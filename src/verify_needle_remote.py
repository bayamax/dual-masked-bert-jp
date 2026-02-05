
import torch
import numpy as np
import os
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from hyperprompt import HyperPromptNet
from hippocampus_v2 import HippocampusV2
import torch.nn as nn

# Define Projector (Must match training)
class RandomProjector(nn.Module):
    def __init__(self, in_dim=2048, out_dim=512): # TinyLlama hidden is 2048
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        torch.manual_seed(42)
        nn.init.orthogonal_(self.proj.weight)
        self.proj.requires_grad_(False)
    def forward(self, x):
        return self.proj(x)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="checkpoints_v2_large/student_last")
    parser.add_argument("--hypernet_path", type=str, default="checkpoints_v2_large/hypernet_last.pt")
    parser.add_argument("--needle", type=str, default="BLUE_KB42")
    parser.add_argument("--distractor_len", type=int, default=2000)
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Running Needle Test v2 (Device: {device}) ===")
    
    # 1. Load Models
    print("Loading Models...")
    # Base
    base_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    base_model = AutoModelForCausalLM.from_pretrained(
        base_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(base_id)
    
    # LoRA Student
    student = PeftModel.from_pretrained(base_model, args.model_path)
    student.eval()
    
    # HyperNet
    hypernet = HyperPromptNet(d_model=2048, num_virtual=1, output_dim=512).to(device).float()
    hypernet.load_state_dict(torch.load(args.hypernet_path, map_location=device))
    hypernet.eval()
    
    # Projector (Fixed)
    projector = RandomProjector(2048, 512).to(device).to(base_model.dtype)
    
    # 2. Setup Dummy Hippocampus (RAM Only)
    hippo = HippocampusV2(storage_dir="temp_hippo_needle")
    # We won't use load(), just add directly
    
    # 3. Construct Data
    print(f"Constructing Haystack (Length: {args.distractor_len} tokens)...")
    needle_text = f"The secret verification code is {args.needle}. "
    filler_text = "The quick brown fox jumps over the lazy dog. " * (args.distractor_len // 10)
    query_text = "What is the secret verification code? The code is "
    
    full_text = needle_text + filler_text + query_text
    
    tokens = tokenizer.encode(full_text, return_tensors="pt").to(device)
    print(f"Total Sequence Length: {tokens.size(1)}")
    
    # 4. Process into Memory (Simulate Phase 0)
    print("Processing into Memory (Chunk=128)...")
    chunk_size = 128
    
    # We scan through the text, generating Z for each chunk and adding to Hippo
    # Note: In real inference, we would do this online. Here we pre-fill.
    
    for i in range(0, tokens.size(1) - chunk_size, chunk_size):
        chunk = tokens[:, i:i+chunk_size]
        if chunk.size(1) != chunk_size: break
        
        with torch.no_grad():
            outputs = student.get_base_model()(chunk, output_hidden_states=True) 
            # Use base model for Z-gen (Teacher) or Student?
            # Training used Teacher (Llama-3-8B) for *Targets* and *Z*.
            # But at Inference, the *Student* must generate Z-Queries.
            # The *Keys* in bank come from Teacher.
            # HERE IS A MISMATCH POTENTIAL.
            # In Phase 0, we used Teacher (Llama-3 or TinyLlama depending on flag).
            # The script `prep_phase0_v2.py` used `--use_tiny` for Large Scale run?
            # Let's check `run_v2_pipeline.py`.
            # `cmd = f"python3 src/prep_phase0_v2.py ... --use_tiny"`
            # YES. We used TinyLlama as Teacher for Phase 0.
            # So Student and Teacher are effectively the same architecture (base TinyLlama).
            # So we can use `student.get_base_model()` or just `base_model` to generate Key Zs.
            
            hiddens = outputs.hidden_states[-1] # [1, 128, 2048]
            chunk_h = hiddens.mean(dim=1) # [1, 2048]
            z_vec = projector(chunk_h).squeeze(0) # [512]
            
            hippo.add(z_vec.float(), chunk.squeeze(0))
            
    print(f"Memory Bank Size: {len(hippo.z_vectors)} chunks.")
    
    # 5. Perform Retrieval at the End
    print("Performing Retrieval...")
    # The last part is the query: "The code is "
    # We take the *last chunk* of the sequence as the "Current Context"
    last_chunk_start = tokens.size(1) - chunk_size
    current_chunk = tokens[:, last_chunk_start:]
    
    # a) Generate Query Z
    with torch.no_grad():
        outputs_c = student(current_chunk, output_hidden_states=True)
        hidden_c = outputs_c.hidden_states[-1] # [1, 128, 2048]
        
        # HyperNet generates Query from Mean Hidden
        z_query = hypernet(hidden_c.float().mean(dim=1).unsqueeze(1)).squeeze(1) # [1, 512]
        
    # b) Search
    # Temporary finalize to stack Z
    hippo.finalize() # Saves to disk
    hippo.load(device=device) # Loads to z_bank
    
    scores, indices = hippo.search(z_query, top_k=3)
    
    print("\n=== Retrieval Results ===")
    top_idx = indices[0, 0].item()
    top_score = scores[0, 0].item()
    print(f"Top-1 Index: {top_idx}, Score: {top_score:.4f}")
    
    retrieved_tokens = hippo.get_text(top_idx).to(device)
    retrieved_str = tokenizer.decode(retrieved_tokens)
    print(f"Retrieved Text: '{retrieved_str}'")
    
    # 6. Verify Generation (Conditioned on Retrieval)
    # We construct input as [Target_KV] + [Current]
    # Re-run Target to get KV
    print("\n=== Generation Test ===")
    
    target_ids = retrieved_tokens.unsqueeze(0)
    
    # Pass 1: Target KV
    with torch.no_grad():
        # Expand pos ids
        t_pos = torch.arange(0, target_ids.size(1), device=device).unsqueeze(0)
        out_past = student(target_ids, position_ids=t_pos, use_cache=True)
        past_kv = out_past.past_key_values
        
        # Pass 2: Current + Gen
        # Next loop to generate tokens
        generated = []
        curr_ids = current_chunk
        
        # We need to shift position ids for current
        c_pos_start = target_ids.size(1)
        
        # Greedy generation for 10 tokens
        for _ in range(10):
            c_pos = torch.arange(c_pos_start, c_pos_start + curr_ids.size(1), device=device).unsqueeze(0)
            
            out_gen = student(curr_ids, position_ids=c_pos, past_key_values=past_kv, use_cache=True)
            next_token_logits = out_gen.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            
            generated.append(next_token.item())
            
            # Update for next step
            past_kv = out_gen.past_key_values
            curr_ids = next_token
            c_pos_start += 1
            
    gen_text = tokenizer.decode(generated)
    print(f"Generated Output: '{gen_text}'")
    
    if args.needle in retrieved_str:
        print("\n[SUCCESS] Needle Retrieved correctly!")
    else:
        print("\n[FAILURE] Needle NOT Retrieved.")
        
    if args.needle in gen_text:
        print("[SUCCESS] Needle Generated correctly!")
    else:
        print("[FAILURE] Needle NOT Generated.")

if __name__ == "__main__":
    main()
