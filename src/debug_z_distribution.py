
import torch
import torch.nn.functional as F
import numpy as np

DATA_FILE = "phase7_cot_chunks.pt"

def main():
    print(f"Loading {DATA_FILE}...")
    samples = torch.load(DATA_FILE)
    
    # Collect first 1000 z-vectors
    z_list = []
    count = 0
    for s in samples:
        for c in s['chunks']:
            z_list.append(c['z_vector'])
            count += 1
            if count >= 1000: break
        if count >= 1000: break
        
    z_tensor = torch.stack(z_list).float()
    z_norm = F.normalize(z_tensor, p=2, dim=1)
    
    print(f"Collected {len(z_list)} Z-vectors.")
    
    # Compute pairwise cosine similarity
    print("Computing pairwise similarity...")
    sim_matrix = torch.mm(z_norm, z_norm.t())
    
    # Mask diagonal
    mask = torch.eye(len(z_list)).bool()
    sim_matrix.masked_fill_(mask, 0.0)
    
    # Statistics
    avg_sim = sim_matrix.mean().item()
    max_sim = sim_matrix.max().item()
    min_sim = sim_matrix.min().item()
    std_sim = sim_matrix.std().item()
    
    print("-" * 30)
    print(f"Average Cosine Sim: {avg_sim:.4f}")
    print(f"Max Cosine Sim:     {max_sim:.4f}")
    print(f"Min Cosine Sim:     {min_sim:.4f}")
    print(f"Std Dev:            {std_sim:.4f}")
    print("-" * 30)
    
    if avg_sim > 0.9:
        print("CRITICAL: High similarity detected! Embeddings are collapsed.")
    elif avg_sim < 0.1:
        print("Embeddings are very orthogonal (good or random).")
    else:
        print("Embeddings show healthy distribution.")

if __name__ == "__main__":
    main()
