import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Configuration matches training
STUDENT_MODEL_PATH = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
HYPERNET_DIM = 2048
LORA_ADAPTER = "phase7_lora_epoch2" 
HYPERNET_WEIGHTS = "phase7_hypernet_epoch2.pt"
DATA_FILE = "phase7_attention_distill.jsonl"
CHUNK_SIZE = 127 # must match training

class HyperNetHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.proj(x)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Load Model
    print("Loading Base Model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL_PATH, 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        attn_implementation="eager"
    )
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_PATH)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading LoRA Adapter from {LORA_ADAPTER}...")
    model = PeftModel.from_pretrained(base_model, LORA_ADAPTER)
    
    print(f"Loading HyperNet Head from {HYPERNET_WEIGHTS}...")
    hypernet = HyperNetHead(model.config.hidden_size, HYPERNET_DIM).to(device).float()
    hypernet.load_state_dict(torch.load(HYPERNET_WEIGHTS, map_location=device))
    hypernet.eval()
    model.eval()

    # 2. Build Tiny Index from Data
    print("Building Retrieval Index from generated samples...")
    chunks = []
    chunk_embeds = []
    
    with open(DATA_FILE, 'r') as f:
        for line in f:
            data = json.loads(line)
            chunks.append(data['top_ref_text']) # We act as if these are the available knowledge
    
    # Embed Chunks in batches to avoid OOM
    embed_batch_size = 100
    all_embeds = []
    
    with torch.no_grad():
        for i in range(0, len(chunks), embed_batch_size):
            batch_chunks = chunks[i:i+embed_batch_size]
            inputs_ref = tokenizer(batch_chunks, return_tensors="pt", padding=True, truncation=True, max_length=CHUNK_SIZE).to(device)
            bs = len(batch_chunks)
            z_prev_dummy = torch.zeros(bs, 1, model.config.hidden_size, device=device).bfloat16()
            
            ref_embeds = model.get_base_model().model.embed_tokens(inputs_ref.input_ids)
            combined_embeds = torch.cat([z_prev_dummy, ref_embeds], dim=1)
            
            out_ref = model.get_base_model()(inputs_embeds=combined_embeds, output_hidden_states=True)
            last_hidden = out_ref.hidden_states[-1][:, -1, :]
            z_refs = hypernet(last_hidden.float()) # [B, Dim]
            z_refs = F.normalize(z_refs, p=2, dim=1)
            all_embeds.append(z_refs.cpu())
            
            del out_ref, ref_embeds, combined_embeds, inputs_ref
            torch.cuda.empty_cache()
        
        chunk_embeds = torch.cat(all_embeds, dim=0).to(device)

    # 3. Test Inference
    print("-" * 50)
    # Get actual question from raw data (which has the original GSM8K question)
    RAW_FILE = "phase7_raw_cot.jsonl"
    test_query = "How many clips did Natalia sell?"  # Fallback
    
    if os.path.exists(RAW_FILE):
        with open(RAW_FILE, 'r') as f:
            first_raw = json.loads(f.readline())
            test_query = first_raw.get('question', test_query)

    print(f"Test Query: {test_query}")
    
    # Embed Query
    with torch.no_grad():
        inputs_q = tokenizer(test_query, return_tensors="pt").to(device)
        outputs_q = model.get_base_model()(inputs_q.input_ids, attention_mask=inputs_q.attention_mask, output_hidden_states=True)
        last_hidden_q = outputs_q.hidden_states[-1][:, -1, :]
        z_q = hypernet(last_hidden_q.float())
        z_q = F.normalize(z_q, p=2, dim=1)
        
        # Retrieve
        scores = torch.mm(z_q, chunk_embeds.t())
        best_idx = torch.argmax(scores).item()
        best_score = scores[0, best_idx].item()
        
        retrieved_text = chunks[best_idx]
        print(f"Retrieval Score: {best_score:.4f}")
        print(f"Retrieved Chunk: {retrieved_text[:100]}...")
        
        # 4. Generate with Injection
        print("Generating Response with Logic Injection...")
        
        # Format: [参照情報: ...]\nUser: ...
        input_text = f"[参照情報: {retrieved_text}]\nUser: {test_query}\nModel:"
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        
        gen_ids = model.generate(
            inputs.input_ids,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7
        )
        output = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        print("-" * 50)
        print("FINAL OUTPUT:")
        print(output)
        print("-" * 50)

if __name__ == "__main__":
    main()
