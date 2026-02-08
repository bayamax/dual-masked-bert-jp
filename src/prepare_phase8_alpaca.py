"""
Phase 8: Reasoning Alpaca Chunk Preparation

Processes DeepSeek's CoT outputs for Alpaca into chunks for training.
Each chunk is 127 tokens, and we store:
- Raw token IDs
- z-vector (computed via HyperNet)
- Chunk index within the CoT

Based on prepare_phase7_cot_chunks.py but adapted for Alpaca dataset format.
"""

import torch
import torch.nn as nn
import json
import os
import argparse
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

# Configuration
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# Use Phase 7.5 Milestone Model
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

def chunk_tokens(token_ids, chunk_size=CHUNK_SIZE):
    """Split token IDs into chunks of chunk_size."""
    chunks = []
    for i in range(0, len(token_ids), chunk_size):
        chunk = token_ids[i:i+chunk_size]
        if len(chunk) > 10:  # Skip very short chunks
            chunks.append(chunk)
    return chunks

class ChunkDataset(Dataset):
    def __init__(self, flat_chunks, tokenizer):
        self.flat_chunks = flat_chunks
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.flat_chunks)
    
    def __getitem__(self, idx):
        item = self.flat_chunks[idx]
        chunk_ids = item['token_ids']
        # Pad
        padded_ids = chunk_ids + [self.tokenizer.pad_token_id] * (CHUNK_SIZE - len(chunk_ids))
        return {
            'input_ids': torch.tensor(padded_ids, dtype=torch.long),
            'original_idx': idx
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="phase8_reasoning_alpaca.jsonl")
    parser.add_argument("--output_file", type=str, default="phase8_alpaca_chunks.pt")
    parser.add_argument("--max_samples", type=int, default=-1)
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Load Tokenizer & Model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager"
    )
    
    is_peft = False
    if os.path.exists(LORA_PATH):
        print(f"Loading LoRA from {LORA_PATH}...")
        model = PeftModel.from_pretrained(base_model, LORA_PATH)
        is_peft = True
    else:
        print(f"WARNING: LoRA not found at {LORA_PATH}. Using base model.")
        model = base_model
    model.eval()
    
    hypernet = HyperNetHead(model.config.hidden_size, HYPERNET_DIM).to(device).float()
    if os.path.exists(HYPERNET_PATH):
        print(f"Loading HyperNet from {HYPERNET_PATH}...")
        hypernet.load_state_dict(torch.load(HYPERNET_PATH, map_location=device))
    else:
        print(f"WARNING: HyperNet not found at {HYPERNET_PATH}. Initializing random.")
        
    hypernet.eval()
    
    # Helper for forward
    def get_forward_model():
        return model.get_base_model() if is_peft else model
        
    def get_embed_layer():
        return model.get_base_model().model.embed_tokens if is_peft else model.model.embed_tokens

    # 2. Pre-process Data (Tokenize & Chunk in CPU)
    print(f"Reading {args.input_file}...")
    
    all_samples_meta = [] # To reconstruct structure
    flat_chunks = []      # For DataLoader
    
    sample_count = 0
    with open(args.input_file, 'r') as f:
        for line in f:
            if args.max_samples > 0 and sample_count >= args.max_samples: break
            try:
                data = json.loads(line)
                
                # Check for valid <think>
                if not data.get('has_think', False) and "<think>" not in data.get('generated_cot', ""):
                     continue
                
                instruction = data['instruction']
                inp = data.get('input', "")
                cot = data['generated_cot']
                
                # Format Question similar to GSM8K training but with Instruction+Input
                if inp.strip():
                    question_text = f"{instruction}\nInput: {inp}"
                else:
                    question_text = instruction
                
                cot_tokens = tokenizer.encode(cot, add_special_tokens=False)
                if len(cot_tokens) < CHUNK_SIZE: continue
                chunks = chunk_tokens(cot_tokens, CHUNK_SIZE)
                if len(chunks) < 2: continue
                
                sample_meta = {
                    'question': question_text, # Unified field name for training script compatibility
                    'full_cot': cot,
                    'num_chunks': len(chunks),
                    'chunk_indices': [] # Will fill with indices into flat_chunks
                }
                
                for i, chunk in enumerate(chunks):
                    # Add to flat list
                    flat_idx = len(flat_chunks)
                    flat_chunks.append({
                        'sample_idx': sample_count,
                        'chunk_index': i,
                        'token_ids': chunk
                    })
                    sample_meta['chunk_indices'].append(flat_idx)
                
                all_samples_meta.append(sample_meta)
                sample_count += 1
                
            except Exception as e:
                pass
                
    print(f"Prepared {len(action := flat_chunks)} chunks from {len(all_samples_meta)} samples.")
    
    # 3. Batch Processing
    dataset = ChunkDataset(flat_chunks, tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    z_vectors_map = {} # flat_idx -> z_tensor
    
    print("Computing Embeddings...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(device)
            indices = batch['original_idx'].tolist()
            
            # Embed
            embeds = get_embed_layer()(input_ids)
            
            # Add Zero Z-Prev
            z_prev_embed = torch.zeros(len(indices), 1, model.config.hidden_size, device=device).bfloat16()
            combined = torch.cat([z_prev_embed, embeds], dim=1)
            
            # Forward
            outputs = get_forward_model()(inputs_embeds=combined, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1][:, -1, :] # [B, Hidden]
            
            # HyperNet
            z = hypernet(last_hidden.float()) # [B, Dim]
            z = z.cpu()
            
            for i, idx in enumerate(indices):
                z_vectors_map[idx] = z[i]
                
    # 4. Reconstruct & Save
    print("Reconstructing Samples...")
    final_output = []
    
    for meta in all_samples_meta:
        chunk_data = []
        for flat_idx in meta['chunk_indices']:
            original_chunk = flat_chunks[flat_idx]
            z = z_vectors_map[flat_idx]
            
            chunk_data.append({
                'chunk_index': original_chunk['chunk_index'],
                'token_ids': original_chunk['token_ids'],
                'z_vector': z
            })
            
        final_output.append({
            'question': meta['question'],
            'full_cot': meta['full_cot'],
            'chunks': chunk_data,
            'num_chunks': meta['num_chunks']
        })
        
    print(f"Saving {len(final_output)} samples to {args.output_file}...")
    torch.save(final_output, args.output_file)
    
    # Stats
    total_chunks = len(flat_chunks)
    print(f"Total chunks processed: {total_chunks}")
    print(f"Avg chunks per sample: {total_chunks/len(final_output):.2f}")

if __name__ == "__main__":
    main()
