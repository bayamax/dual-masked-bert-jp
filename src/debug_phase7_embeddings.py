"""
Debug Phase 7 Embeddings

Checks:
1. Diversity of chunk embeddings (are they all the same?)
2. Content of the raw data (is it all the same?)
3. HyperNet output range
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
import numpy as np

# Configuration
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_PATH = "phase7_revised_lora_epoch0"
HYPERNET_PATH = "phase7_revised_hypernet_epoch0.pt"
DATA_FILE = "phase7_raw_cot.jsonl"
HYPERNET_DIM = 2048
CHUNK_SIZE = 127

class HyperNetHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.proj(x)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
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
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model.eval()
    
    # Load HyperNet
    hypernet = HyperNetHead(model.config.hidden_size, HYPERNET_DIM).to(device).float()
    hypernet.load_state_dict(torch.load(HYPERNET_PATH, map_location=device))
    hypernet.eval()
    
    # Check Data
    print("\nChecking Data...")
    chunks = []
    with open(DATA_FILE, 'r') as f:
        for i, line in enumerate(f):
            if i >= 20: break
            data = json.loads(line)
            chunks.append(data['generated_cot'][:100])
            print(f"Sample {i}: {data['generated_cot'][:50]}...")
    
    # Check Embeddings for 10 samples
    print("\nChecking Embeddings...")
    inputs = tokenizer(chunks[:10], return_tensors="pt", padding=True, truncation=True, max_length=CHUNK_SIZE).to(device)
    
    with torch.no_grad():
        bs = len(chunks[:10])
        z_prev_dummy = torch.zeros(bs, 1, model.config.hidden_size, device=device).bfloat16()
        ref_embeds = model.get_base_model().model.embed_tokens(inputs.input_ids)
        combined_embeds = torch.cat([z_prev_dummy, ref_embeds], dim=1)
        
        out = model.get_base_model()(inputs_embeds=combined_embeds, output_hidden_states=True)
        last_hidden = out.hidden_states[-1][:, -1, :]
        z = hypernet(last_hidden.float())
        z_norm = F.normalize(z, p=2, dim=1)
        
        print("\nEmbedding Stats:")
        print(f"Mean: {z.mean().item():.4f}")
        print(f"Std: {z.std().item():.4f}")
        print(f"Min: {z.min().item():.4f}")
        print(f"Max: {z.max().item():.4f}")
        
        # Calculate pairwise cosine similarity
        sim_matrix = torch.mm(z_norm, z_norm.t())
        print("\nSimilarity Matrix (first 5x5):")
        print(sim_matrix[:5, :5])
        
        # Check if collapsed
        off_diag = sim_matrix - torch.eye(bs, device=device)
        avg_sim = off_diag.sum() / (bs * (bs - 1))
        print(f"\nAverage Pairwise Similarity: {avg_sim.item():.4f}")
        
        if avg_sim > 0.99:
            print("WARNING: Embeddings have collapsed to a single point!")
        else:
            print("Embeddings show diversity.")

if __name__ == "__main__":
    main()
