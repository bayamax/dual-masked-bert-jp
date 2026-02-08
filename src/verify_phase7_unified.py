"""
Phase 7 Unified Verification (Retrieval Fix + DeepSeek Format)

Verifies:
1. Retrieval: Can we find the correct chunk (Old Z) using the Question (New Z)?
2. Generation: Does the model generate valid <think>...<ref>... structure?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
import os

# Configuration
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# Path to the NEWLY trained model
LORA_PATH = "phase7_retrieval_fix_epoch0/lora_epoch0" 
HYPERNET_PATH = "phase7_retrieval_fix_epoch0/hypernet_epoch0.pt"

# Path to PRECOMPUTED chunks (Old Z space)
DATA_PT_FILE = "phase7_cot_chunks.pt" 
# Raw text for readable reference (optional, but good for checking content)
DATA_JSONL_FILE = "phase7_raw_cot.jsonl"

HYPERNET_DIM = 2048
CHUNK_SIZE = 127

# Tags
USER_TAG = "<user>"
USER_END = "</user>\n"
MODEL_TAG = "<model>"
THINK_TAG = "<think>\n"
REF_TAG = "<ref>"
REF_END = "</ref>"

class HyperNetHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.proj(x)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Load Model & HyperNet
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="eager"
    )
    
    print(f"Loading LoRA from {LORA_PATH}...")
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model.eval()
    
    # Load HyperNet
    # Note: Training script saved state_dict directly to .pt file
    print(f"Loading HyperNet from {HYPERNET_PATH}...")
    hypernet = HyperNetHead(model.config.hidden_size, HYPERNET_DIM).to(device).float()
    hypernet.load_state_dict(torch.load(HYPERNET_PATH, map_location=device))
    hypernet.eval()
    
    # 2. Build Index from Precomputed Zs
    print("Loading Data & Building Index...")
    if not os.path.exists(DATA_PT_FILE):
        print(f"Error: {DATA_PT_FILE} not found. Needs prepared chunks.")
        return
        
    samples = torch.load(DATA_PT_FILE)
    
    # For verification, let's index ALL chunks from the first 50 samples
    index_z = []
    index_meta = []
    
    count = 0
    # Limit samples for speed
    limit_samples = 50
    
    for i, sample in enumerate(samples):
        if i >= limit_samples: break
        
        question = sample['question']
        chunks = sample['chunks']
        
        for j, chunk in enumerate(chunks):
            # Use the Z-vector stored in the file (Old Space)
            z = chunk['z_vector'].float() # Ensure float
            index_z.append(z)
            index_meta.append({
                'sample_idx': i,
                'chunk_idx': j,
                'question': question,
                # Decode safely roughly
                'text': tokenizer.decode(chunk['token_ids'], skip_special_tokens=True)
            })
            count += 1
            
    index_tensor = torch.stack(index_z).to(device)
    # Normalize index
    index_tensor = F.normalize(index_tensor, p=2, dim=1)
    
    print(f"Indexed {count} chunks from {limit_samples} samples.")
    
    # 3. Test Retrieval & Generation
    print("\n" + "="*60)
    print("VERIFICATION: Retrieval & Generation")
    print("="*60)
    
    test_samples = samples[:5]
    
    for i, sample in enumerate(test_samples):
        q = sample['question']
        print(f"\nQuery {i}: {q[:50]}...")
        
        # --- A. RETRIEVAL ---
        # Encode Query using NEW HyperNet
        q_inputs = tokenizer(q, return_tensors="pt", padding=True, truncation=True, max_length=CHUNK_SIZE).to(device)
        
        with torch.no_grad():
            out = model.get_base_model()(q_inputs.input_ids, output_hidden_states=True)
            last = out.hidden_states[-1][:, -1, :]
            z_q = hypernet(last.float())
            z_q = F.normalize(z_q, p=2, dim=1)
            
        # Dot Product
        # z_q: [1, dim] x index: [N, dim].T -> [1, N]
        scores = torch.mm(z_q, index_tensor.t()).squeeze() # [N]
        top_k = torch.topk(scores, k=3)
        
        # Find correct chunk: Sample i, Chunk 0
        target_idx_in_index = -1
        for idx, meta in enumerate(index_meta):
            if meta['sample_idx'] == i and meta['chunk_idx'] == 0:
                target_idx_in_index = idx
                break
        
        print("  Top 3 Retrieval:")
        for rank, idx in enumerate(top_k.indices.tolist()):
            meta = index_meta[idx]
            is_correct = (idx == target_idx_in_index)
            status = "[CORRECT]" if is_correct else ""
            print(f"    {rank+1}. Score={top_k.values[rank]:.4f} | S{meta['sample_idx']}-C{meta['chunk_idx']} {status} | {meta['text'][:30]}...")
        
        if target_idx_in_index != -1:
            target_score = scores[target_idx_in_index].item()
            target_rank_global = (scores > target_score).sum().item() + 1
            print(f"  Result: Correct Chunk Rank = {target_rank_global} / {count} (Score {target_score:.4f})")
        else:
            print("  Result: Correct Chunk not indexed (Sample > limit?)")

        # --- B. GENERATION ---
        # Prompt: <user>Q</user><model><think><ref>
        # We manually inject <ref> tag to prompt retrieval/generation
        prompt = f"{USER_TAG}{q}{USER_END}{MODEL_TAG}\n{THINK_TAG}{REF_TAG}"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        print("  Generating...")
        gen_out = model.generate(
            **inputs, 
            max_new_tokens=40, 
            do_sample=False, 
            pad_token_id=tokenizer.pad_token_id
        )
        gen_text = tokenizer.decode(gen_out[0], skip_special_tokens=False)
        # Show output starting from prompt end
        generated_part = gen_text[len(prompt):].replace("\n", " ")
        print(f"  Gen Output: {generated_part[:100]}...")

if __name__ == "__main__":
    main()
