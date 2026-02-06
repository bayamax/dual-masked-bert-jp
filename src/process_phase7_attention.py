import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import json
import os
from tqdm import tqdm
import argparse

# Reuse Configuration
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
INPUT_FILE = "phase7_raw_cot.jsonl"
OUTPUT_FILE = "phase7_attention_distill.jsonl"
MAX_LENGTH = 2048
RECENT_WINDOW = 64
CHUNK_SIZE = 32

def main():
    print(f"Loading Teacher for Analysis: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # We use HF model here because we need 'output_attentions=True', 
    # which vLLM doesn't easily expose in python API for individual tokens? 
    # vLLM is for generation. For analysis we need raw forward pass with attentions.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        torch_dtype=torch.bfloat16, 
        load_in_4bit=True,
        device_map="auto", 
        attn_implementation="eager" 
    )
    
    print(f"Processing data from {INPUT_FILE}...")
    
    start_idx = 0
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r') as f_check:
            start_idx = sum(1 for _ in f_check)
            
    print(f"Resuming processing from index {start_idx}...")
    
    with open(INPUT_FILE, 'r') as f_in, open(OUTPUT_FILE, 'a') as f_out:
        lines = f_in.readlines()
        
        # Slicing lines from start_idx
        for line in tqdm(lines[start_idx:]):
            try:
                data = json.loads(line)
            except: continue
            
            prompt = data['prompt']
            cot = data['generated_cot']
            
            # Reconstruct Full Text
            full_text = prompt + cot
            
            # Tokenize
            tokens = tokenizer.encode(full_text)
            
            # Truncate
            if len(tokens) > MAX_LENGTH:
                tokens = tokens[:MAX_LENGTH]
            
            input_ids = torch.tensor([tokens]).to(model.device)
            seq_len = input_ids.shape[1]
            
            # Skip short
            if seq_len < RECENT_WINDOW + CHUNK_SIZE: continue
            
            # Forward Pass
            with torch.no_grad():
                outputs = model(input_ids, output_attentions=True)
            
            # Attention Extraction (Same Logic)
            attentions = outputs.attentions[-4:] 
            avg_attn = torch.stack(attentions).mean(dim=(0, 2)) 
            final_attn_dist = avg_attn[0, -1, :] 
            final_attn_dist[0] = 0.0 
            
            recent_start_idx = seq_len - RECENT_WINDOW
            
            attn_recent = final_attn_dist[recent_start_idx:]
            attn_past = final_attn_dist[1:recent_start_idx]
            
            sum_recent = attn_recent.sum().item()
            sum_past = attn_past.sum().item()
            total_attn = sum_recent + sum_past + 1e-9
            
            ratio_ref = sum_past / total_attn
            ratio_query = sum_recent / total_attn
            
            # Save Logic (Relaxed for now, save all valid length)
            past_len = len(attn_past)
            num_chunks = (past_len + CHUNK_SIZE - 1) // CHUNK_SIZE
            
            chunk_scores = []
            chunk_texts = []
            past_tokens = input_ids[0, 1:recent_start_idx]
            
            valid_chunks = True
            for k in range(num_chunks):
                start = k * CHUNK_SIZE
                end = min((k + 1) * CHUNK_SIZE, past_len)
                score = attn_past[start:end].sum().item()
                chunk_scores.append(score)
                decoded = tokenizer.decode(past_tokens[start:end], skip_special_tokens=True)
                if len(decoded.strip()) == 0: 
                    # valid_chunks = False; break 
                    # Instead of breaking, just append empty? Or skip empty chunks?
                    chunk_texts.append("") # Keep alignment with index
                else:
                    chunk_texts.append(decoded)
            
            if valid_chunks:
                total_chunk_score = sum(chunk_scores) + 1e-9
                chunk_probs = [s / total_chunk_score for s in chunk_scores]
                
                # Best ref text: pick argmax
                best_idx = np.argmax(chunk_probs)
                best_ref_text = chunk_texts[best_idx]
                
                query_text = tokenizer(tokenizer.decode(input_ids[0, recent_start_idx:], skip_special_tokens=True), add_special_tokens=False).input_ids # Re-decode/encode safety?
                # Just usage decoded string
                query_text = tokenizer.decode(input_ids[0, recent_start_idx:], skip_special_tokens=True)
                
                sample_entry = {
                    "query_text": query_text,
                    "top_ref_text": best_ref_text,
                    "label_chunk_probs": chunk_probs,
                    "label_attn_ratio_ref": ratio_ref,
                    "label_attn_ratio_query": ratio_query
                }
                
                f_out.write(json.dumps(sample_entry) + "\n")
            
            # Cleanup
            del outputs
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
