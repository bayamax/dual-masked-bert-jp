"""
Phase 7 Unified Verification

Tests the unified model (HyperNet + LoRA trained together)
for both retrieval and generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json

# Configuration
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_PATH = "phase7_unified_lora_epoch0"
HYPERNET_PATH = "phase7_unified_hypernet_epoch0.pt"
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
    
    # Build index from raw data
    print("Building index...")
    chunks = []
    questions_list = []
    answers_list = []
    
    with open(DATA_FILE, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                cot = data['generated_cot'][:300]
                chunks.append(cot)
                questions_list.append(data['question'])
            except:
                continue
    
    # Embed chunks in batches
    embed_batch_size = 100
    all_embeds = []
    
    with torch.no_grad():
        for i in range(0, min(len(chunks), 1000), embed_batch_size):  # Limit to 1000 for speed
            batch_chunks = chunks[i:i+embed_batch_size]
            inputs_ref = tokenizer(batch_chunks, return_tensors="pt", padding=True, truncation=True, max_length=CHUNK_SIZE).to(device)
            bs = len(batch_chunks)
            z_prev_dummy = torch.zeros(bs, 1, model.config.hidden_size, device=device).bfloat16()
            
            ref_embeds = model.get_base_model().model.embed_tokens(inputs_ref.input_ids)
            combined_embeds = torch.cat([z_prev_dummy, ref_embeds], dim=1)
            
            out_ref = model.get_base_model()(inputs_embeds=combined_embeds, output_hidden_states=True)
            last_hidden = out_ref.hidden_states[-1][:, -1, :]
            z_refs = hypernet(last_hidden.float())
            z_refs = F.normalize(z_refs, p=2, dim=1)
            all_embeds.append(z_refs.cpu())
            
            del out_ref, ref_embeds, combined_embeds, inputs_ref
            torch.cuda.empty_cache()
        
        chunk_embeds = torch.cat(all_embeds, dim=0).to(device)
    
    print(f"Indexed {chunk_embeds.shape[0]} chunks")
    
    # Test
    print("\n" + "=" * 60)
    print("INTEGRATED TEST (Unified Model)")
    print("=" * 60)
    
    test_questions = questions_list[:5]
    
    for i, question in enumerate(test_questions, 1):
        # Embed query
        with torch.no_grad():
            inputs_q = tokenizer(question, return_tensors="pt").to(device)
            outputs_q = model.get_base_model()(inputs_q.input_ids, output_hidden_states=True)
            last_hidden_q = outputs_q.hidden_states[-1][:, -1, :]
            z_q = hypernet(last_hidden_q.float())
            z_q = F.normalize(z_q, p=2, dim=1)
            
            # Retrieve
            scores = torch.mm(z_q, chunk_embeds.t())
            best_idx = torch.argmax(scores).item()
            best_score = scores[0, best_idx].item()
            retrieved = chunks[best_idx]
        
        # Generate
        input_text = f"<ref>{retrieved}</ref>\nQuestion: {question}\nAnswer:"
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
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
        print(f"Retrieved: {retrieved[:60]}...")
        print(f"Answer: {response}")
        print("-" * 40)
    
    print("\n" + "=" * 60)
    print("Test Complete!")

if __name__ == "__main__":
    main()
