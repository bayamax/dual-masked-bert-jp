"""
Phase 7.6 Output Verification: Base vs Correction Accuracy

Objective:
1. Verify if the "1% Base Accuracy" seen in training logs is real or a reporting bug.
2. Measure the "Correction Capability": If we give the model a WRONG chunk, can it find the CORRECT one?

Metrics:
- Base Top-1 Recall: Q -> Correct Chunk (Z similarity)
- Correction Top-1 Recall: Q + Mistake -> Correct Chunk
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os
import random
import numpy as np

# Config
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATA_FILE = "phase7_cot_chunks.pt"
# Models to Compare
MODELS_TO_TEST = [
    ("Base (Phase 7.5)", "phase7_accuracy_boost_best/lora", "phase7_accuracy_boost_best/hypernet.pt"),
    ("Corrected (Phase 7.6)", "phase7_online_correction_epoch2", "phase7_online_correction_epoch2/hypernet.pt")
]

device = "cuda" if torch.cuda.is_available() else "cpu"
CHUNK_SIZE = 127
TEST_SAMPLES = 500 # Speed up verification

# Tags
USER_TAG = "<user>"
USER_END = "</user>\n"
MODEL_TAG = "<model>"
THINK_TAG = "<think>\n"
REF_TAG = "<ref>"
REF_END = "</ref>"

class HyperNetHead(nn.Module):
    def __init__(self, input_dim=2048, output_dim=2048):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.proj(x)

def load_data():
    print(f"Loading {DATA_FILE}...")
    all_samples = torch.load(DATA_FILE)
    # Filter valid
    all_samples = [s for s in all_samples if s['chunks'] and s['chunks'][0].get('z_vector') is not None]
    
    # Select subset
    random.seed(42)
    random.shuffle(all_samples)
    subset = all_samples[:TEST_SAMPLES]
    
    # Build Global Index (Z vectors) from the subset (or full set? For speed, subset index is easier but less realistic. Let's use subset index)
    # Strictly speaking, retrieval should be against ALL chunks. 
    # But for debugging "1% vs 40%", a subset index of 500*AverageChunks is fine.
    
    index_z = []
    index_map = [] # (sample_idx, chunk_idx)
    
    print("Building Index...")
    for s_i, s in enumerate(subset):
        for c_i, c in enumerate(s['chunks']):
            z = c['z_vector'].float()
            index_z.append(z)
            index_map.append((s_i, c_i))
            
    index_tensor = torch.stack(index_z).to(device)
    index_tensor = F.normalize(index_tensor, p=2, dim=1)
    
    return subset, index_tensor, index_map

def evaluate_model(name, lora_path, hypernet_path, subset, index_tensor, index_map):
    print(f"\n--- Evaluating {name} ---")
    if not os.path.exists(lora_path):
        print(f"Skipping {name}: path not found.")
        return
        
    # Load Model
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, device_map=device)
    model = PeftModel.from_pretrained(base, lora_path)
    model.eval()
    
    hypernet = HyperNetHead().to(device)
    # HyperNet might be state_dict or full model? 
    # Previous script saved: torch.save(hypernet.state_dict(), ...)
    hn_state = torch.load(hypernet_path, map_location=device)
    hypernet.load_state_dict(hn_state)
    hypernet.eval()
    
    # --- DYNAMIC RE-EMBEDDING ---
    print(f"Re-Embedding {len(subset)} samples (x {len([c for s in subset for c in s['chunks']])} chunks) for {name}...")
    
    # We need to rebuild index_tensor and index_map for THIS model
    # Because z-vectors in file might be stale
    
    current_index_z = []
    current_index_map = []
    
    chunk_batch_size = 32
    all_chunk_tokens = []
    
    for s_i, sample in enumerate(subset):
        for c_i, c in enumerate(sample['chunks']):
            # Use token_ids directly
            all_chunk_tokens.append(c['token_ids'])
            
    # Batch Process
    pad_id = tokenizer.pad_token_id
    
    with torch.no_grad():
        for i in range(0, len(all_chunk_tokens), chunk_batch_size):
            batch_tokens = all_chunk_tokens[i : i+chunk_batch_size]
            
            # Manual Padding
            max_len = max(len(t) for t in batch_tokens)
            if max_len > CHUNK_SIZE: max_len = CHUNK_SIZE # Cap at 127
            
            input_ids_list = []
            attention_mask_list = []
            
            for t in batch_tokens:
                t = t[:max_len] # Truncate
                pad_len = max_len - len(t)
                padded = t + [pad_id] * pad_len
                mask = [1] * len(t) + [0] * pad_len
                input_ids_list.append(padded)
                attention_mask_list.append(mask)
            
            input_ids = torch.tensor(input_ids_list, device=device)
            attention_mask = torch.tensor(attention_mask_list, device=device)
            
            # Forward Base
            out = model.get_base_model()(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            
            # Gather Last Real Token
            seq_lens = attention_mask.sum(dim=1) - 1
            last_hidden_states = out.hidden_states[-1]
            batch_h = last_hidden_states[torch.arange(len(batch_tokens), device=device), seq_lens, :]
            
            # HyperNet
            batch_z = hypernet(batch_h.float())
            batch_z = F.normalize(batch_z, p=2, dim=1)
            
            current_index_z.append(batch_z)
            
    # Stacking
    if current_index_z:
        index_tensor_dynamic = torch.cat(current_index_z, dim=0)
    else:
        index_tensor_dynamic = torch.empty((0, 2048), device=device)
        
    print(f"Re-Embedded Index Shape: {index_tensor_dynamic.shape}")
    
    # Update map
    # We need to separate index_map into index_map_dynamic corresponding to all_chunk_meta
    index_map_dynamic = all_chunk_meta
    
    # NOW use index_tensor_dynamic for search
    index_tensor = index_tensor_dynamic
    index_map = index_map_dynamic
    
    correct_base = 0
    soft_base = 0
    correct_correction = 0
    soft_correction = 0
    total = 0
    
    for s_i, sample in enumerate(subset):
        q_text = sample['question']
        
        # 1. Base Retrieval
        # Input: Q
        input_text = f"{USER_TAG}{q_text}{USER_END}{MODEL_TAG}\n{THINK_TAG}{REF_TAG}"
        inputs = tokenizer(input_text, return_tensors="pt", add_special_tokens=False).to(device)
        
        with torch.no_grad():
            out = model.get_base_model()(input_ids=inputs.input_ids, output_hidden_states=True)
            t_last = out.hidden_states[-1][:, -1, :]
            z_q = hypernet(t_last.float())
            z_q = F.normalize(z_q, p=2, dim=1)
            
            # Search
            sim = torch.mm(z_q, index_tensor.t())
            top_val, top_idx = torch.topk(sim, k=1)
            retrieved_global_idx = top_idx.item()
            
            # Check correctness
            r_s_i, r_c_i = index_map[retrieved_global_idx]
            
            is_correct_base = (r_s_i == s_i and r_c_i == 0) # Must find Chunk 0 of correct sample
            is_soft_base = (r_s_i == s_i) # Any chunk from correct sample
            
            if is_correct_base:
                correct_base += 1
            if is_soft_base:
                soft_base += 1
            
            # 2. Correction Retrieval
            # Scenario: If correct, we test distractor. If incorrect, we use retrieved as distractor.
            
            if is_correct_base:
                # Force mistake: Pick a random WRONG chunk from index
                while True:
                    dist_idx = random.randint(0, len(index_map)-1)
                    d_s_i, d_c_i = index_map[dist_idx]
                    if d_s_i != s_i or d_c_i != 0:
                        break
                distractor_global_idx = dist_idx
            else:
                distractor_global_idx = retrieved_global_idx
                
            # Get Distractor Tokens
            # We need the tokens of the Mistake Chunk.
            # In training we had them. Here we need to reconstruct or look up?
            # subset[d_s_i]['chunks'][d_c_i]['token_ids'] ?
            d_s_i, d_c_i = index_map[distractor_global_idx]
            distractor_chunk = subset[d_s_i]['chunks'][d_c_i]
            dist_tokens = distractor_chunk.get('token_ids') 
            if dist_tokens is None:
                # Recalculate if not cached (should be cached in .pt)
                dist_text = distractor_chunk['text']
                dist_tokens = tokenizer(dist_text, add_special_tokens=False)['input_ids']
            
            # Formulate Correction Input
            # Context: Q + <think><ref>WRONG_CHUNK</ref>
            # Note: The model expects the WRONG chunk INSIDE the first ref tag? 
            # Wait, training logic:
            # prefix_text = f"{USER_TAG}{q_text}{USER_END}{MODEL_TAG}\n{THINK_TAG}{REF_TAG}"
            # context_ids = prefix_ids + distractor_tokens + suffix_ids
            # suffix_ids = REF_END (encode(REF_END)) -> "</ref>"
            # So the input is: Q ... <ref>WRONG_TOKENS</ref>
            # And we extract Z at the end of this sequence.
            
            ref_end_ids = tokenizer.encode(REF_END, add_special_tokens=False)
            
            # Careful with context concatenation
            # inputs.input_ids is [prefix]
            # correction_input = prefix + dist + ref_end
            
            corr_input_ids = torch.cat([
                inputs.input_ids[0],
                torch.tensor(dist_tokens, device=device),
                torch.tensor(ref_end_ids, device=device)
            ]).unsqueeze(0)
            
            # Forward
            out_corr = model.get_base_model()(input_ids=corr_input_ids, output_hidden_states=True)
            t_last_corr = out_corr.hidden_states[-1][:, -1, :]
            z_corr = hypernet(t_last_corr.float())
            z_corr = F.normalize(z_corr, p=2, dim=1)
            
            # Search
            sim_corr = torch.mm(z_corr, index_tensor.t())
            top_val_c, top_idx_c = torch.topk(sim_corr, k=1)
            retrieved_c_global = top_idx_c.item()
            
            rc_s_i, rc_c_i = index_map[retrieved_c_global]
            
            # Correction Metrics
            # Strict: Same Document AND Chunk 0
            if rc_s_i == s_i and rc_c_i == 0:
                correct_correction += 1
            
            # Soft: Same Document (Any Chunk)
            if rc_s_i == s_i:
                soft_correction += 1
                
        total += 1
        if total % 100 == 0:
            print(f"Processed {total}...")
            
    print(f"\nResults for {name}:")
    print(f"Base Strict Top-1: {correct_base}/{total} ({correct_base/total:.2%})")
    print(f"Base Soft Top-1 (Doc): {soft_base}/{total} ({soft_base/total:.2%})")
    print(f"Correction Strict Top-1: {correct_correction}/{total} ({correct_correction/total:.2%})")
    print(f"Correction Soft Top-1: {soft_correction}/{total} ({soft_correction/total:.2%})")

if __name__ == "__main__":
    subset, index_tensor, index_map = load_data()
    for name, lora, hn in MODELS_TO_TEST:
        evaluate_model(name, lora, hn, subset, index_tensor, index_map)
