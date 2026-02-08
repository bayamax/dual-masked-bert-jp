
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import random
import os

# Configuration
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_PATH = "phase7_accuracy_boost_best/lora" 
HYPERNET_PATH = "phase7_accuracy_boost_best/hypernet.pt"
DATA_PT_FILE = "phase7_cot_chunks.pt" 
HYPERNET_DIM = 2048
CHUNK_SIZE = 127

# Tags
USER_TAG = "<user>"
USER_END = "</user>\n"
MODEL_TAG = "<model>"
THINK_TAG = "<think>\n"
REF_TAG = "<ref>"
REF_END = "</ref>"

class HyperNetHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.proj(x)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Load Data
    print("Loading Data...")
    all_samples = torch.load(DATA_PT_FILE)
    all_samples = [s for s in all_samples if s['num_chunks'] >= 2]
    
    # Val Split (Seed 42)
    random.seed(42)
    random.shuffle(all_samples)
    val_len = int(len(all_samples) * 0.1)
    val_samples = all_samples[len(all_samples)-val_len:]
    distractor_samples = all_samples[:1000]
    index_samples = val_samples + distractor_samples
    
    # 2. Build Index
    # We need tokenizer first to decode text if 'text' key is missing
    print("Loading Model & Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    print("Building Index...")
    index_z = []
    index_meta = []
    
    for i, sample in enumerate(index_samples):
        for j, chunk in enumerate(sample['chunks']):
            z = chunk['z_vector'].float()
            index_z.append(z)
            
            # Decode text from token_ids if 'text' not in chunk
            text = chunk.get('text', tokenizer.decode(chunk['token_ids'], skip_special_tokens=True))
            
            index_meta.append({
                'orig_question': sample['question'],
                'chunk_idx': j,
                'text': text, 
                'full_cot': sample.get('full_cot', '')
            })
            
    index_tensor = torch.stack(index_z).to(device)
    index_tensor = F.normalize(index_tensor, p=2, dim=1)
    
    # 3. Load Model
    
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="eager"
    )
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model.eval()
    
    hypernet = HyperNetHead(model.config.hidden_size, HYPERNET_DIM).to(device).float()
    hypernet.load_state_dict(torch.load(HYPERNET_PATH, map_location=device))
    hypernet.eval()
    
    # 4. Generation Experiment
    print("\n=== Jump-Start Generation Experiment ===")
    print("Method: Q -> Retrieve Chunk -> Feed as Context -> Generate Continuation")
    
    test_count = 0
    for i, sample in enumerate(val_samples):
        # We look for samples where retrieval is "Soft Correct" (same Q) but maybe not Chunk 0
        q = sample['question']
        
        # Retrieval
        q_ids = tokenizer.encode(q, add_special_tokens=False)
        q_ids = q_ids[:CHUNK_SIZE]
        q_ids += [tokenizer.pad_token_id] * (CHUNK_SIZE - len(q_ids))
        q_input_ids = torch.tensor([q_ids], device=device)
        
        with torch.no_grad():
            out = model.get_base_model()(input_ids=q_input_ids, output_hidden_states=True)
            last = out.hidden_states[-1][:, -1, :]
            z_q = F.normalize(hypernet(last.float()), p=2, dim=1)
            
        scores = torch.mm(z_q, index_tensor.t()).squeeze()
        best_idx = torch.argmax(scores).item()
        best_meta = index_meta[best_idx]
        
        # Only test if it retrieved a chunk from the CORRECT question (Soft Match)
        if best_meta['orig_question'] != q:
            continue
            
        test_count += 1
        print(f"\n[Sample {i}] Soft Match Found!")
        print(f"Q: {q[:100]}...")
        print(f"Retrieved: Chunk {best_meta['chunk_idx']} (Score: {scores[best_idx]:.4f})")
        print(f"Chunk Text: {best_meta['text'][:100]}...")
        
        # Generate Continuation
        # Prompt: <ref>{Retrieved Chunk}</ref>
        # The model should generate the NEXT chunk (or answer)
        
        prompt_text = f"{REF_TAG}{best_meta['text']}{REF_END}"
        input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
        
        print("Generating continuation...")
        with torch.no_grad():
            gen_ids = model.generate(
                input_ids,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id
            )
            
        output_text = tokenizer.decode(gen_ids[0][len(input_ids[0]):], skip_special_tokens=True)
        print(f"Generated: {output_text}")
        
        # Check against Full CoT
        # Ideally we see if the generation overlaps with subsequent chunks
        target_cot_start = best_meta['text'][-20:] # overlap check
        print(f"Ground Truth (Next Chunks approximation): ...")
        
        if test_count >= 5: # Test 5 samples
            break

if __name__ == "__main__":
    main()
