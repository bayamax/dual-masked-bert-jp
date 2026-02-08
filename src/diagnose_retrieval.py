"""
Diagnostic script to investigate retrieval issue.
Checks:
1. Whether chunk embeddings are diverse
2. Sample of different chunks in the data
3. Retrieval with different queries
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json

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
    
    # Load raw data
    print("\n=== Checking raw data ===")
    chunks = []
    questions = []
    
    with open(DATA_FILE, 'r') as f:
        for i, line in enumerate(f):
            if i >= 20:  # Check first 20
                break
            try:
                data = json.loads(line)
                cot = data['generated_cot'][:300]
                chunks.append(cot)
                questions.append(data['question'])
            except:
                continue
    
    print(f"Loaded {len(chunks)} chunks")
    print("\n--- Sample chunks (first 100 chars) ---")
    for i, chunk in enumerate(chunks[:5]):
        print(f"[{i}]: {chunk[:100]}...")
    
    # Check if chunks are different
    print("\n=== Checking chunk diversity ===")
    unique_prefixes = set(c[:50] for c in chunks)
    print(f"Unique prefixes (first 50 chars): {len(unique_prefixes)} / {len(chunks)}")
    
    # Embed chunks and check diversity
    print("\n=== Checking embedding diversity ===")
    all_embeds = []
    
    with torch.no_grad():
        for i, chunk in enumerate(chunks[:10]):
            inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=CHUNK_SIZE).to(device)
            z_prev_dummy = torch.zeros(1, 1, model.config.hidden_size, device=device).bfloat16()
            
            embeds = model.get_base_model().model.embed_tokens(inputs.input_ids)
            combined = torch.cat([z_prev_dummy, embeds], dim=1)
            
            out = model.get_base_model()(inputs_embeds=combined, output_hidden_states=True)
            last_hidden = out.hidden_states[-1][:, -1, :]
            z = hypernet(last_hidden.float())
            z = F.normalize(z, p=2, dim=1)
            all_embeds.append(z)
    
    all_embeds = torch.cat(all_embeds, dim=0)
    
    # Compute pairwise similarities
    print("\n--- Pairwise cosine similarities ---")
    sim_matrix = torch.mm(all_embeds, all_embeds.t())
    
    print("Similarity matrix (first 5x5):")
    for i in range(min(5, sim_matrix.shape[0])):
        row = [f"{sim_matrix[i, j].item():.3f}" for j in range(min(5, sim_matrix.shape[1]))]
        print(f"  {row}")
    
    # Check off-diagonal similarities
    off_diag = []
    for i in range(sim_matrix.shape[0]):
        for j in range(sim_matrix.shape[1]):
            if i != j:
                off_diag.append(sim_matrix[i, j].item())
    
    avg_off_diag = sum(off_diag) / len(off_diag) if off_diag else 0
    max_off_diag = max(off_diag) if off_diag else 0
    min_off_diag = min(off_diag) if off_diag else 0
    
    print(f"\nOff-diagonal similarities:")
    print(f"  Avg: {avg_off_diag:.4f}")
    print(f"  Max: {max_off_diag:.4f}")
    print(f"  Min: {min_off_diag:.4f}")
    
    if avg_off_diag > 0.9:
        print("\n⚠️ WARNING: Embeddings are very similar! HyperNet may be collapsing.")
    elif avg_off_diag > 0.7:
        print("\n⚠️ CAUTION: Embeddings have moderate similarity.")
    else:
        print("\n✓ Embeddings show good diversity.")
    
    # Test retrieval with different queries
    print("\n=== Testing retrieval ===")
    test_queries = [
        "Natalia sold clips",
        "Calculate the total cost",
        "How many pages"
    ]
    
    chunk_embeds = all_embeds  # Use the 10 we already embedded
    
    for query in test_queries:
        with torch.no_grad():
            inputs_q = tokenizer(query, return_tensors="pt").to(device)
            z_prev_q = torch.zeros(1, 1, model.config.hidden_size, device=device).bfloat16()
            embeds_q = model.get_base_model().model.embed_tokens(inputs_q.input_ids)
            combined_q = torch.cat([z_prev_q, embeds_q], dim=1)
            out_q = model.get_base_model()(inputs_embeds=combined_q, output_hidden_states=True)
            last_q = out_q.hidden_states[-1][:, -1, :]
            z_q = hypernet(last_q.float())
            z_q = F.normalize(z_q, p=2, dim=1)
            
            scores = torch.mm(z_q, chunk_embeds.t())
            best_idx = torch.argmax(scores).item()
            best_score = scores[0, best_idx].item()
            
            print(f"\nQuery: '{query}'")
            print(f"  Best match: idx={best_idx}, score={best_score:.4f}")
            print(f"  All scores: {[f'{s:.3f}' for s in scores[0].tolist()]}")

if __name__ == "__main__":
    main()
