"""
Phase 7 Revised: CoT Chunk Preparation

Processes DeepSeek's CoT outputs into chunks for training.
Each chunk is 127 tokens, and we store:
- Raw token IDs
- z-vector (computed via HyperNet)
- Chunk index within the CoT
"""

import torch
import torch.nn as nn
import json
import os
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Configuration
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_PATH = "phase7_unified_lora_epoch0"
HYPERNET_PATH = "phase7_unified_hypernet_epoch0.pt"
CHUNK_SIZE = 127
HYPERNET_DIM = 2048

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="phase7_raw_cot.jsonl")
    parser.add_argument("--output_file", type=str, default="phase7_cot_chunks.pt")
    parser.add_argument("--max_samples", type=int, default=-1)
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load tokenizer and model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager"
    )
    
    # Load LoRA if exists
    if os.path.exists(LORA_PATH):
        print(f"Loading LoRA from {LORA_PATH}...")
        model = PeftModel.from_pretrained(base_model, LORA_PATH)
    else:
        print("No LoRA found, using base model")
        model = base_model
    model.eval()
    
    # Load HyperNet if exists
    hypernet = HyperNetHead(model.config.hidden_size, HYPERNET_DIM).to(device).float()
    if os.path.exists(HYPERNET_PATH):
        print(f"Loading HyperNet from {HYPERNET_PATH}...")
        hypernet.load_state_dict(torch.load(HYPERNET_PATH, map_location=device))
    hypernet.eval()
    
    # Process CoT data
    print(f"Processing {args.input_file}...")
    
    all_samples = []
    sample_count = 0
    
    with open(args.input_file, 'r') as f:
        for line in f:
            if args.max_samples > 0 and sample_count >= args.max_samples:
                break
            
            try:
                data = json.loads(line)
                question = data['question']
                cot = data['generated_cot']
                
                # Tokenize the full CoT
                cot_tokens = tokenizer.encode(cot, add_special_tokens=False)
                
                if len(cot_tokens) < CHUNK_SIZE:
                    continue
                
                # Split into chunks
                chunks = chunk_tokens(cot_tokens, CHUNK_SIZE)
                
                if len(chunks) < 2:
                    continue
                
                # Compute z-vector for each chunk
                chunk_data = []
                z_prev = torch.zeros(1, HYPERNET_DIM, device=device)
                
                with torch.no_grad():
                    for i, chunk_ids in enumerate(chunks):
                        # Pad chunk to CHUNK_SIZE
                        padded_ids = chunk_ids + [tokenizer.pad_token_id] * (CHUNK_SIZE - len(chunk_ids))
                        input_ids = torch.tensor([padded_ids], device=device)
                        
                        # Get embeddings
                        embeds = model.get_base_model().model.embed_tokens(input_ids)
                        
                        # Add z_prev as prefix
                        z_prev_embed = torch.zeros(1, 1, model.config.hidden_size, device=device).bfloat16()
                        combined = torch.cat([z_prev_embed, embeds], dim=1)
                        
                        # Forward pass
                        outputs = model.get_base_model()(inputs_embeds=combined, output_hidden_states=True)
                        last_hidden = outputs.hidden_states[-1][:, -1, :]
                        
                        # Compute z
                        z = hypernet(last_hidden.float())
                        
                        chunk_data.append({
                            'chunk_index': i,
                            'token_ids': chunk_ids,
                            'z_vector': z.cpu().squeeze(0),
                        })
                        
                        z_prev = z.detach()
                
                all_samples.append({
                    'question': question,
                    'full_cot': cot,
                    'chunks': chunk_data,
                    'num_chunks': len(chunks)
                })
                
                sample_count += 1
                if sample_count % 100 == 0:
                    print(f"Processed {sample_count} samples...")
                    
            except Exception as e:
                print(f"Error: {e}")
                continue
    
    print(f"Total samples: {len(all_samples)}")
    
    # Save
    torch.save(all_samples, args.output_file)
    print(f"Saved to {args.output_file}")
    
    # Stats
    total_chunks = sum(s['num_chunks'] for s in all_samples)
    print(f"Total chunks: {total_chunks}")
    print(f"Avg chunks per sample: {total_chunks / len(all_samples):.1f}")

if __name__ == "__main__":
    main()
