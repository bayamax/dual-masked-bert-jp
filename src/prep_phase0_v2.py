
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from tqdm import tqdm
from hippocampus_v2 import HippocampusV2
import torch.nn as nn

class RandomProjector(nn.Module):
    def __init__(self, in_dim=4096, out_dim=512):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        # Fix weights for consistency
        torch.manual_seed(42)
        nn.init.orthogonal_(self.proj.weight)
        self.proj.requires_grad_(False) # Frozen

    def forward(self, x):
        return self.proj(x)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="wiki_long_50k.jsonl")
    parser.add_argument("--save_dir", type=str, default="hippocampus_v2_data")
    parser.add_argument("--chunk_size", type=int, default=128) # B=128
    parser.add_argument("--threshold", type=float, default=0.05) # Attention Threshold
    parser.add_argument("--use_tiny", action="store_true", help="Fallback to TinyLlama")
    parser.add_argument("--max_docs", type=int, default=-1, help="Limit number of docs for demo")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Teacher
    if args.use_tiny:
        model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        hidden_dim = 2048
    else:
        model_id = "unsloth/llama-3-8b-Instruct-bnb-4bit"
        hidden_dim = 4096
        
    print(f"Loading Teacher: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map="auto", 
        output_attentions=True,
        trust_remote_code=True,
        output_hidden_states=True
    )
    model.eval()
    
    # 2. Projector (Z-Gen)
    projector = RandomProjector(hidden_dim, 512).to(device)
    
    # 3. Hippocampus
    hippo = HippocampusV2(d_model=512, storage_dir=args.save_dir)
    
    # 4. Process Data
    print("Processing Data...")
    with open(args.data_path, 'r') as f:
        lines = f.readlines()
        
        # Limit for demo/testing if overly large, but Phase 0 implies full pass.
        # We will process linearly.
        
        doc_count = 0 
        total_chunks = 0
        
        for line in tqdm(lines):
            item = json.loads(line)
            text = item["text"]
            tokens = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt").to(device)
            seq_len = tokens.size(1)
            
            if seq_len < args.chunk_size * 2:
                continue
                
            # Split into chunks
            chunks = []
            for i in range(0, seq_len, args.chunk_size):
                chunk = tokens[:, i:i+args.chunk_size]
                if chunk.size(1) == args.chunk_size:
                    chunks.append(chunk)
                    
            if len(chunks) < 2: continue
            
            # --- Encoding & Z-Generation ---
            # We need to process sequential chunks to get hidden states AND attention
            # Ideally, process entire sequence at once?
            # Llama-3-8B might OOM on 20k context if not careful.
            # But "wiki_long" usually fits in 8k or 4k.
            # Let's try full sequence forward pass (with no grad).
            
            try:
                with torch.no_grad():
                    outputs = model(tokens)
                    # Last Hidden State for Z
                    # [1, Seq, Dim]
                    hiddens = outputs.hidden_states[-1]
                    
                    # Attention: tuple of [1, Heads, Seq, Seq]
                    # We usually look at last layer or average of last few.
                    # Llama3 has 32 layers. Let's look at Layer 30.
                    # outputs.attentions is a tuple.
                    attn_map = outputs.attentions[-1].mean(dim=1).squeeze(0) # [Seq, Seq]
                    
            except torch.cuda.OutOfMemoryError:
                print("OOM, skipping doc")
                continue
                
            # Process Chunks
            doc_chunk_indices = []
            
            for i, chunk in enumerate(chunks):
                # 1. Extract Representative Hidden State for this chunk
                # (e.g., Mean or Last Token)
                # Let's use Mean of the chunk's hidden states
                start = i * args.chunk_size
                end = start + args.chunk_size
                
                chunk_h = hiddens[:, start:end, :].mean(dim=1) # [1, Dim]
                z_vec = projector(chunk_h).squeeze(0) # [512]
                
                # 2. Add to Bank (Everything gets an ID)
                # Note: We store the TOKEN IDS for reconstruction
                bank_idx = hippo.add(z_vec, chunk.squeeze(0))
                doc_chunk_indices.append(bank_idx)
                
                # 3. Labeling (If i > 0)
                # Check attention from Current (i) to Past (j < i)
                if i > 0:
                    # Current Query Range: start:end
                    # Max attention to which past block?
                    
                    # Only look at past, excluding self
                    best_j = -1
                    max_score = 0.0
                    
                    for j in range(i):
                        past_s = j * args.chunk_size
                        past_e = past_s + args.chunk_size
                        
                        # Sum attention from [start:end] to [past_s:past_e]
                        # Submatrix: Rows (Queries) = Current, Cols (Keys) = Past
                        sub_attn = attn_map[start:end, past_s:past_e]
                        
                        # Total attention mass
                        score = sub_attn.sum().item()
                        # Normalize by chunk size? 
                        # Or simple sum. If strong link, sum is high.
                        
                        if score > max_score:
                            max_score = score
                            best_j = j
                            
                    # Threshold Check
                    # Normalize score? Theoretical max is chunk_size (if all attention goes there).
                    # Threshold 0.05 means 5% of attention mass goes to that block.
                    norm_score = max_score / args.chunk_size
                    
                    if norm_score > args.threshold and best_j != -1:
                        # Found a significant link!
                        target_bank_id = doc_chunk_indices[best_j]
                        hippo.add_label(bank_idx, target_bank_id)
                        
            doc_count += 1
            if args.max_docs > 0 and doc_count >= args.max_docs:
                break
                
            if doc_count % 100 == 0:
                print(f"Processed {doc_count} docs. Bank size: {len(hippo.z_vectors)}")
                
    hippo.finalize()
    print("Done. Phase 0 Complete.")

if __name__ == "__main__":
    main()
