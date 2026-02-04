
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os
from tqdm import tqdm
from hippocampus_v2 import HippocampusV2
import torch.nn as nn
import torch.nn.functional as F

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
    projector.to(dtype=model.dtype)
    
    # 3. Hippocampus
    hippo = HippocampusV2(d_model=512, storage_dir=args.save_dir)
    # Don't call load() if we are appending writing. The init sets up paths.
    # But if we want to add_label to existing indices, we might need to know count?
    # hippo.add() returns correct index (len of list).
    # If we restart process, self.z_vectors is empty.
    # We should load existing Z-vectors to append correctly?
    # Or just start a new chunk?
    # HippocampusV2 logic: `self.z_vectors = []`. `finalize` -> `torch.save`.
    # If we blindly `finalize` again, we overwrite `z_vectors.bin`.
    # CRITICAL: We need RESUME MODE in HippocampusV2 or here.
    # If we want to append, we should load existing.
    
    if os.path.exists(hippo.z_path):
        print("Loading existing Hippocampus state for append...")
        hippo.load() 
        # load() puts z_vectors into self.z_bank (Tensor).
        # We need them in self.z_vectors (List) for appending?
        # Or convert back.
        # Efficient way: keep old Z on disk, write new Z to "z_vectors_part2.bin"?
        # Complexity: High.
        # Simple way: Load all to RAM (List), append, save all.
        # 40k docs -> 100k chunks? 50MB. Safe to load all.
        if hasattr(hippo, 'z_bank'):
            hippo.z_vectors = [z.cpu().to(torch.int8) for z in hippo.z_bank]
            # text_offsets already loaded.
            # text_file opened as 'r'. Close it and reopen 'a+'.
            hippo.text_file.close()
            hippo.text_file = open(hippo.text_path, 'a+') # Append mode
            print(f"Loaded {len(hippo.z_vectors)} existing vectors.")
            
    # 4. Resume Check
    progress_file = os.path.join(args.save_dir, "phase0_progress.txt")
    resume_idx = 0
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                resume_idx = int(f.read().strip())
        except:
            pass
            
    print(f"Resuming from Input Line Index: {resume_idx}")

    # 5. Process Data
    print("Processing Data...")
    with open(args.data_path, 'r') as f:
        doc_count = 0 
        
        # Linear Scan to skip
        # If resume_idx is huge, this takes a few seconds. fast.
        
        pbar = tqdm(total=50000)
        pbar.update(resume_idx)
        
        for line_idx, line in enumerate(f):
            if line_idx < resume_idx:
                continue
                
            try:
                item = json.loads(line)
                text = item["text"]
                tokens = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt").to(device)
                seq_len = tokens.size(1)
                
                if seq_len < args.chunk_size * 2:
                    pbar.update(1)
                    continue
                    
                # Split into chunks
                chunks = []
                for i in range(0, seq_len, args.chunk_size):
                    chunk = tokens[:, i:i+args.chunk_size]
                    if chunk.size(1) == args.chunk_size:
                        chunks.append(chunk)
                        
                if len(chunks) < 2: 
                    pbar.update(1)
                    continue
                
                # --- Encoding & Z-Generation ---
                try:
                    with torch.no_grad():
                        outputs = model(tokens)
                        hiddens = outputs.hidden_states[-1]
                        attn_map = outputs.attentions[-1].mean(dim=1).squeeze(0)
                        
                except torch.cuda.OutOfMemoryError:
                    print(f"OOM at doc {line_idx}, skipping")
                    del tokens, outputs, hiddens, attn_map
                    torch.cuda.empty_cache()
                    pbar.update(1)
                    continue
                    
                # Process Chunks
                doc_chunk_indices = []
                
                for i, chunk in enumerate(chunks):
                    start = i * args.chunk_size
                    end = start + args.chunk_size
                    
                    chunk_h = hiddens[:, start:end, :].mean(dim=1) 
                    z_vec = projector(chunk_h).squeeze(0) 
                    
                    bank_idx = hippo.add(z_vec, chunk.squeeze(0))
                    doc_chunk_indices.append(bank_idx)
                    
                    if i > 0:
                        best_j = -1
                        max_score = 0.0
                        
                        for j in range(i):
                            past_s = j * args.chunk_size
                            past_e = past_s + args.chunk_size
                            
                            sub_attn = attn_map[start:end, past_s:past_e]
                            score = sub_attn.sum().item()
                            
                            if score > max_score:
                                max_score = score
                                best_j = j
                                
                        norm_score = max_score / args.chunk_size
                        
                        if norm_score > args.threshold and best_j != -1:
                            target_bank_id = doc_chunk_indices[best_j]
                            hippo.add_label(bank_idx, target_bank_id)
                            
                doc_count += 1
                
                # Save Progress periodically
                if doc_count % 100 == 0:
                    with open(progress_file, 'w') as pf:
                        pf.write(str(line_idx + 1))
                    if doc_count % 1000 == 0:
                        # Periodic finalize (save Z) just in case?
                        # HippocampusV2 finalize writes all.
                        # We can call finalize multiple times if we reload?
                        # No, finalize closes file.
                        pass
                        
                if args.max_docs > 0 and doc_count >= args.max_docs:
                    break
                    
                pbar.update(1)
                
            except Exception as e:
                print(f"Error at line {line_idx}: {e}")
                continue
                
    hippo.finalize()
    # Save final progress
    with open(progress_file, 'w') as pf:
        pf.write(str(line_idx + 1))
        
    print("Done. Phase 0 Complete.")

if __name__ == "__main__":
    main()
