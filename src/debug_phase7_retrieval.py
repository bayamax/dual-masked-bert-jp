"""
Debug Phase 7 Retrieval Detail

Checks:
1. Retrieval scores for test queries
2. Top-5 retrieved chunks for each query
3. Whether query embeddings are degenerate
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
    
    # helper for embeddings
    def get_z(text):
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=CHUNK_SIZE).to(device)
        with torch.no_grad():
            ref_embeds = model.get_base_model().model.embed_tokens(inputs.input_ids)
            # Query has no z_prev context, retrieval chunks DO have z_prev context (usually)
            # BUT for simple index building we used zero vector. Consistent? Yes.
            bs = inputs.input_ids.shape[0]
            z_prev_dummy = torch.zeros(bs, 1, model.config.hidden_size, device=device).bfloat16()
            combined_embeds = torch.cat([z_prev_dummy, ref_embeds], dim=1)
            
            out = model.get_base_model()(inputs_embeds=combined_embeds, output_hidden_states=True)
            last_hidden = out.hidden_states[-1][:, -1, :]
            z = hypernet(last_hidden.float())
            return F.normalize(z, p=2, dim=1)

    # 1. Index 100 chunks
    print("\nBuilding validation index (100 chunks)...")
    chunks = []
    questions_list = []
    
    with open(DATA_FILE, 'r') as f:
        for i, line in enumerate(f):
            if i >= 100: break
            data = json.loads(line)
            chunks.append(data['generated_cot'][:200]) # First 200 chars
            questions_list.append(data['question'])
    
    chunk_embeds = get_z(chunks)
    
    # 2. Test Retrieval
    test_questions = questions_list[:5]
    
    print("\nRetrieval Analysis:")
    
    for i, q in enumerate(test_questions):
        print(f"\nQuery {i}: {q[:50]}...")
        
        # Embed query
        # NOTE: Verification script splits query logic from chunk logic.
        # Query: [Query Tokens] -> Last Hidden -> HyperNet
        # Chunk: [z_prev] [Chunk Tokens] -> Last Hidden -> HyperNet
        # Are they aligned?
        
        # Standard query embedding (as done in verify script)
        inputs_q = tokenizer(q, return_tensors="pt").to(device)
        with torch.no_grad():
            out_q = model.get_base_model()(inputs_q.input_ids, output_hidden_states=True)
            last_hidden_q = out_q.hidden_states[-1][:, -1, :]
            z_q = hypernet(last_hidden_q.float())
            z_q = F.normalize(z_q, p=2, dim=1)
        
        # Scores
        scores = torch.mm(z_q, chunk_embeds.t()).squeeze()
        top_k = torch.topk(scores, k=5)
        
        print(f"Query Z stats: Mean={z_q.mean():.4f} Std={z_q.std():.4f}")
        print("Top 5 matches:")
        for rank, (score, idx) in enumerate(zip(top_k.values, top_k.indices)):
            idx = idx.item()
            print(f"  {rank+1}. Score={score:.4f} | Index={idx} | Content={chunks[idx][:40]}...")
            
        # Is the correct chunk (index i) in top 5?
        target_score = scores[i].item()
        target_rank = (scores > target_score).sum().item() + 1
        print(f"Target Chunk (Index {i}): Rank={target_rank} | Score={target_score:.4f}")

if __name__ == "__main__":
    main()
