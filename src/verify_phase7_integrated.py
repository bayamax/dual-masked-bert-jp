"""
Phase 7 Integrated Verification

End-to-end test:
1. HyperNet (distillation-trained) retrieves relevant chunk from memory
2. SFT LoRA generates answer using the retrieved chunk

This tests the full Phase 7 pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
import os

# Configuration
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DISTILL_LORA = "phase7_lora_epoch2"  # For retrieval embedding
SFT_LORA = "phase7_sft_lora_epoch2"  # For generation
HYPERNET_WEIGHTS = "phase7_hypernet_epoch2.pt"
DATA_FILE = "phase7_attention_distill.jsonl"
RAW_FILE = "phase7_raw_cot.jsonl"
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
    
    # ========== RETRIEVAL SETUP ==========
    print("\n[1/4] Loading Retrieval Model (Distillation LoRA + HyperNet)...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model for retrieval
    retrieval_base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager"
    )
    retrieval_model = PeftModel.from_pretrained(retrieval_base, DISTILL_LORA)
    retrieval_model.eval()
    
    # Load HyperNet
    hypernet = HyperNetHead(retrieval_model.config.hidden_size, HYPERNET_DIM).to(device).float()
    hypernet.load_state_dict(torch.load(HYPERNET_WEIGHTS, map_location=device))
    hypernet.eval()
    
    # ========== BUILD INDEX ==========
    print("\n[2/4] Building Retrieval Index...")
    chunks = []
    with open(DATA_FILE, 'r') as f:
        for line in f:
            data = json.loads(line)
            chunks.append(data['top_ref_text'])
    
    # Embed chunks in batches
    embed_batch_size = 100
    all_embeds = []
    
    with torch.no_grad():
        for i in range(0, len(chunks), embed_batch_size):
            batch_chunks = chunks[i:i+embed_batch_size]
            inputs_ref = tokenizer(batch_chunks, return_tensors="pt", padding=True, truncation=True, max_length=CHUNK_SIZE).to(device)
            bs = len(batch_chunks)
            z_prev_dummy = torch.zeros(bs, 1, retrieval_model.config.hidden_size, device=device).bfloat16()
            
            ref_embeds = retrieval_model.get_base_model().model.embed_tokens(inputs_ref.input_ids)
            combined_embeds = torch.cat([z_prev_dummy, ref_embeds], dim=1)
            
            out_ref = retrieval_model.get_base_model()(inputs_embeds=combined_embeds, output_hidden_states=True)
            last_hidden = out_ref.hidden_states[-1][:, -1, :]
            z_refs = hypernet(last_hidden.float())
            z_refs = F.normalize(z_refs, p=2, dim=1)
            all_embeds.append(z_refs.cpu())
            
            del out_ref, ref_embeds, combined_embeds, inputs_ref
            torch.cuda.empty_cache()
        
        chunk_embeds = torch.cat(all_embeds, dim=0).to(device)
    
    print(f"Indexed {len(chunks)} chunks")
    
    # ========== GENERATION SETUP ==========
    print("\n[3/4] Loading Generation Model (SFT LoRA)...")
    # We need to unload the distillation LoRA and load SFT LoRA
    # Reload base model for generation
    del retrieval_model, retrieval_base
    torch.cuda.empty_cache()
    
    gen_base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    gen_model = PeftModel.from_pretrained(gen_base, SFT_LORA)
    gen_model.eval()
    
    # ========== TEST ==========
    print("\n[4/4] Running Integrated Test...")
    print("=" * 60)
    
    # Get test questions from raw data
    with open(RAW_FILE, 'r') as f:
        test_samples = [json.loads(line) for line in f.readlines()[:5]]
    
    for i, sample in enumerate(test_samples, 1):
        question = sample['question']
        
        # Step 1: Embed query using HyperNet
        with torch.no_grad():
            inputs_q = tokenizer(question, return_tensors="pt").to(device)
            # Use gen_model's base for query embedding (should be similar enough)
            outputs_q = gen_model.get_base_model()(inputs_q.input_ids, output_hidden_states=True)
            last_hidden_q = outputs_q.hidden_states[-1][:, -1, :]
            z_q = hypernet(last_hidden_q.float())
            z_q = F.normalize(z_q, p=2, dim=1)
            
            # Step 2: Retrieve best chunk
            scores = torch.mm(z_q, chunk_embeds.t())
            best_idx = torch.argmax(scores).item()
            best_score = scores[0, best_idx].item()
            retrieved_chunk = chunks[best_idx]
        
        # Step 3: Generate with SFT model
        input_text = f"[参照情報: {retrieved_chunk[:400]}]\n\nUser: {question}\nModel:"
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = gen_model.generate(
                inputs.input_ids,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        
        new_tokens = outputs[0][inputs.input_ids.shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        print(f"\n[Test {i}]")
        print(f"Q: {question[:80]}...")
        print(f"Retrieval Score: {best_score:.4f}")
        print(f"Retrieved: {retrieved_chunk[:60]}...")
        print(f"Answer: {response}")
        print("-" * 40)
    
    print("\n" + "=" * 60)
    print("Integrated Test Complete!")

if __name__ == "__main__":
    main()
