"""
Phase 7.6: Hard Negative Mining

Finds the "Hardest Negatives" for each training sample to train the correction mechanism.
For each Question, we find the chunk that the model *mistakenly* retrieves (highest similarity score) that is NOT the correct answer.

Output: phase7_hard_negatives.pt
List of { 'question': ..., 'target_chunk_idx': ..., 'hard_negative_chunk_idx': ... }
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm
import argparse
import os

# Configuration
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# Use Phase 7.5 Milestone
LORA_PATH = "phase7_accuracy_boost_best/lora"
HYPERNET_PATH = "phase7_accuracy_boost_best/hypernet.pt"
CHUNK_SIZE = 127
HYPERNET_DIM = 2048
BATCH_SIZE = 32

class HyperNetHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.proj(x)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default="phase7_cot_chunks.pt")
    parser.add_argument("--output_file", type=str, default="phase7_hard_negatives.pt")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Load Model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager"
    )
    
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model.eval()
    
    hypernet = HyperNetHead(model.config.hidden_size, HYPERNET_DIM).to(device).float()
    hypernet.load_state_dict(torch.load(HYPERNET_PATH, map_location=device))
    hypernet.eval()
    
    # 2. Load Data & Index Vectors
    print(f"Loading Data: {args.data_file}")
    all_samples = torch.load(args.data_file)
    # Filter for valid samples
    all_samples = [s for s in all_samples if s['num_chunks'] >= 2]
    
    # Pre-compute all Z vectors for global search
    print("Indexing all chunk vectors...")
    all_z_list = []
    chunk_meta_map = [] # idx -> {sample_idx, chunk_index, token_ids}
    
    for s_idx, sample in enumerate(all_samples):
        for c in sample['chunks']:
            all_z_list.append(c['z_vector']) # Already CPU tensor
            chunk_meta_map.append({
                'sample_idx': s_idx,
                'chunk_index': c['chunk_index'],
                'token_ids': c['token_ids']
            })
            
    all_z_matrix = torch.stack(all_z_list).to(device).float() # [TotalChunks, Dim]
    all_z_matrix = F.normalize(all_z_matrix, p=2, dim=1)
    
    print(f"Indexed {len(all_z_list)} chunks.")
    
    # 3. Mine Hard Negatives
    print("Mining Hard Negatives...")
    hard_negatives_data = []
    
    # We process each sample's QUESTION to find what it retrieves
    # Batch processing query generation for speed
    
    questions = [s['question'] for s in all_samples]
    # Simple chunking for tokenizer
    
    for i in tqdm(range(0, len(all_samples), BATCH_SIZE)):
        batch_samples = all_samples[i : i+BATCH_SIZE]
        batch_questions = [s['question'] for s in batch_samples]
        
        # Tokenize Questions
        inputs = tokenizer(batch_questions, return_tensors="pt", padding=True, truncation=True, max_length=CHUNK_SIZE)
        input_ids = inputs.input_ids.to(device)
        
        with torch.no_grad():
            outputs = model.get_base_model()(input_ids=input_ids, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1][:, -1, :] # [B, Hidden]
            z_q = hypernet(last_hidden.float())
            z_q = F.normalize(z_q, p=2, dim=1)
            
            # Search
            # Similarity: [B, TotalChunks]
            sim_scores = torch.mm(z_q, all_z_matrix.t())
            
            # Get Top-K to filter out the correct answer
            # We want the highest scoring chunk that is NOT the target (Chunk 0 of this sample)
            topk_vals, topk_indices = torch.topk(sim_scores, k=50, dim=1)
            
            topk_indices = topk_indices.cpu().numpy()
            
            for b in range(len(batch_samples)):
                sample = batch_samples[b]
                sample_global_idx = i + b
                
                # Identify Correct Answer's global index (Chunk 0)
                # In chunk_meta_map, we need to find which one corresponds to this sample's chunk 0
                # But we don't track global indices directly in 'sample'.
                # Let's rely on Sample Metadata matching.
                
                # Goal: Find Hardest Negative
                hardest_neg_idx = -1
                
                for candidate_idx in topk_indices[b]:
                    meta = chunk_meta_map[candidate_idx]
                    
                    # Hard Negative Definition:
                    # 1. NOT the correct chunk (Start Chunk of THIS sample)
                    # 2. Can be a later chunk of SAME sample (Context Misalignment)
                    # 3. Can be a chunk from DIFFERENT sample (Distractor)
                    
                    # Correct Answer Condition:
                    is_correct_answer = (meta['sample_idx'] == sample_global_idx) and (meta['chunk_index'] == 0)
                    
                    # If it's the correct answer, skip
                    if is_correct_answer: continue
                    
                    # Found the highest scoring non-answer!
                    hardest_neg_idx = candidate_idx
                    break
                
                if hardest_neg_idx != -1:
                    # Store Data
                    neg_meta = chunk_meta_map[hardest_neg_idx]
                    target_chunk = sample['chunks'][0]
                    
                    hard_negatives_data.append({
                        'question': sample['question'],
                        'target_chunk': target_chunk, # Contains token_ids and z_vector
                        'hard_negative_chunk': {
                            'token_ids': neg_meta['token_ids'],
                            'chunk_index': neg_meta['chunk_index'],
                            'from_same_sample': (meta['sample_idx'] == sample_global_idx)
                        }
                    })

    print(f"Mined {len(hard_negatives_data)} hard negative pairs.")
    torch.save(hard_negatives_data, args.output_file)
    print(f"Saved to {args.output_file}")

if __name__ == "__main__":
    main()
