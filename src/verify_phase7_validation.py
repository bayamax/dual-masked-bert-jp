"""
Phase 7 Validation Verification

Target: Check performance specifically on the Held-Out Validation Set.
Replicates the random split logic from training (seed 42) to identify val samples.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import random
import os

# Configuration
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# We will point this to the BEST model after training
LORA_PATH = "phase7_accuracy_boost_best/lora" 
HYPERNET_PATH = "phase7_accuracy_boost_best/hypernet.pt"

DATA_PT_FILE = "phase7_cot_chunks.pt" 
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
    
    # 1. Load Data & Replicate Split
    print("Loading Data & Replicating Split...")
    if not os.path.exists(DATA_PT_FILE):
        print(f"Error: {DATA_PT_FILE} not found.")
        return
        
    all_samples = torch.load(DATA_PT_FILE)
    # Filter same as training
    all_samples = [s for s in all_samples if s['num_chunks'] >= 2]
    
    # Shuffle (Seed 42)
    random.seed(42)
    random.shuffle(all_samples)
    
    total_len = len(all_samples)
    val_len = int(total_len * 0.1)
    train_len = total_len - val_len
    
    val_samples = all_samples[train_len:]
    print(f"Total: {total_len}, Val Set: {len(val_samples)} samples")
    
    # 2. Build Index (Ideally index EVERYTHING or just Val? Realistically we search everything)
    # But for "Top-1 Accuracy on Val", we want to see if Val Question finds Val Chunk among ALL chunks.
    # Indexing 7000+ chunks takes memory.
    # Let's just index the Validation Set's chunks for now to check "Recall within Val Set".
    # (Or we can index a subset of Train + Val to make it harder).
    
    # Let's index ALL validation chunks + 1000 Training chunks as distractors.
    distractor_samples = all_samples[:1000]
    index_samples = val_samples + distractor_samples
    
    print(f"Building Index with {len(index_samples)} samples (Val + 1000 Train distractors)...")
    
    index_z = []
    index_meta = []
    
    count = 0
    for i, sample in enumerate(index_samples):
        question = sample['question']
        chunks = sample['chunks']
        
        # We only strictly care about Chunk 0 for Q->C0
        for j, chunk in enumerate(chunks):
            z = chunk['z_vector'].float()
            index_z.append(z)
            index_meta.append({
                'sample_idx': i, # Relative to index_samples
                'orig_question': question,
                'chunk_idx': j,
                'is_val': (sample in val_samples)
            })
            count += 1
            
    index_tensor = torch.stack(index_z).to(device)
    index_tensor = F.normalize(index_tensor, p=2, dim=1)
    print(f"Indexed {count} chunks.")

    # 3. Load Model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="eager"
    )
    
    if os.path.exists(LORA_PATH):
        print(f"Loading LoRA from {LORA_PATH}...")
        model = PeftModel.from_pretrained(base_model, LORA_PATH)
    else:
        print(f"WARNING: {LORA_PATH} not found. Using base model (will fail retrieval).")
        model = base_model
        
    model.eval()
    
    hypernet = HyperNetHead(model.config.hidden_size, HYPERNET_DIM).to(device).float()
    if os.path.exists(HYPERNET_PATH):
        print(f"Loading HyperNet from {HYPERNET_PATH}...")
        hypernet.load_state_dict(torch.load(HYPERNET_PATH, map_location=device))
    else:
        print("WARNING: HyperNet not found.")
    hypernet.eval()
    
    # 4. Evaluate
    print("\n" + "="*60)
    print("VALIDATION SET EVALUATION")
    print("="*60)
    
    correct_count = 0
    top5_count = 0
    total_eval = 0
    
    # Check first 50 val samples
    eval_target_samples = val_samples[:50]
    
    for i, sample in enumerate(eval_target_samples):
        q = sample['question']
        
        # Encode Q (Match Training Logic: Pad to CHUNK_SIZE)
        q_ids = tokenizer.encode(q, add_special_tokens=False)
        q_ids = q_ids[:CHUNK_SIZE]
        q_ids += [tokenizer.pad_token_id] * (CHUNK_SIZE - len(q_ids))
        q_input_ids = torch.tensor([q_ids], device=device)
        
        with torch.no_grad():
            out = model.get_base_model()(input_ids=q_input_ids, output_hidden_states=True)
            last = out.hidden_states[-1][:, -1, :]
            z_q = hypernet(last.float())
            z_q = F.normalize(z_q, p=2, dim=1)
            
        # Retrieve
        scores = torch.mm(z_q, index_tensor.t()).squeeze()
        top_k = torch.topk(scores, k=5)
        
        # Verify
        # Correct answer is this sample's Chunk 0.
        # We need to find the index of this sample's Chunk 0 in index_meta.
        # Note: sample is in index_samples.
        
        # Find matches
        found_rank = -1
        
        # Optimization: We know where we put it if we tracked strictly, but linear search is safe
        # Ideally we search by string match of question?
        
        for rank, idx in enumerate(top_k.indices.tolist()):
            meta = index_meta[idx]
            if meta['orig_question'] == q and meta['chunk_idx'] == 0:
                found_rank = rank + 1
                break
        
        if found_rank == 1: correct_count += 1
        if found_rank != -1: top5_count += 1
        
        total_eval += 1
        
        if i < 10: # Print details for first 10
            # Get score of the correct chunk
            # Correct chunk index in 'index_tensor' is needed.
            # We iterate to find it.
            correct_idx = -1
            for idx, meta in enumerate(index_meta):
                if meta['orig_question'] == q and meta['chunk_idx'] == 0:
                    correct_idx = idx
                    break
            
            correct_score = scores[correct_idx].item()
            top1_score = top_k.values[0].item()
            top1_text = index_meta[top_k.indices[0].item()]['orig_question'][:30]
            
            print(f"Query {i}: {q[:30]}... | Rank: {found_rank if found_rank!=-1 else '>5'}")
            print(f"  Correct Score: {correct_score:.4f}")
            print(f"  Top-1 Score:   {top1_score:.4f} (Q: {top1_text}...)")
            
    print("-" * 30)
    print(f"Final Results on {total_eval} Validation Samples:")
    print(f"Top-1 Accuracy: {correct_count/total_eval:.2%} ({correct_count}/{total_eval})")
    print(f"Top-5 Accuracy: {top5_count/total_eval:.2%} ({top5_count}/{total_eval})")

if __name__ == "__main__":
    main()
