import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
import json
import os
from tqdm import tqdm

# Configuration
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
DATA_OUTPUT = "phase7_attention_distill.jsonl"
MAX_LENGTH = 4096
RECENT_WINDOW = 256 # Considered as "Query/Prompt" area
CHUNK_SIZE = 127 # set to 127 to allow 1 token for 'z' (Recursive Structure: 127 text + 1 z = 128)
NUM_SAMPLES = 5000

def main():
    print(f"Loading Teacher: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        torch_dtype=torch.bfloat16, 
        device_map="auto", 
        attn_implementation="eager" 
    )
    
    # Check context window
    print(f"Model File: {model.config._name_or_path}")
    
    print("Loading Dataset...")
    dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    
    data_buffer = []
    
    for i, sample in enumerate(tqdm(dataset)):
        if len(data_buffer) >= NUM_SAMPLES: break
        
        text = sample['text']
        tokens = tokenizer(text, return_tensors="pt").input_ids[0].to(model.device)
        
        if len(tokens) < 1000: continue
        
        # Split: [Past] -- [Recent (Query)]
        # We ensure "Past" is modulo 128 for clean chunking
        
        total_len = min(len(tokens), MAX_LENGTH)
        
        # End point logic
        # We want to predict the continuation AFTER the recent window
        # Input = tokens[:split_point], Target = tokens[split_point:] (implicit in generation)
        
        # Actually, for attention map extraction, we feed the sequence and look at the LAST token's attention.
        input_ids = tokens[:total_len].unsqueeze(0)
        
        seq_len = input_ids.shape[1]
        if seq_len < RECENT_WINDOW + CHUNK_SIZE: continue

        # Run Forward Pass
        with torch.no_grad():
            outputs = model(input_ids, output_attentions=True)
        
        # Extract Attention from the last token
        # Averaging across last 4 layers
        attentions = outputs.attentions[-4:] 
        avg_attn = torch.stack(attentions).mean(dim=(0, 2)) # (1, Seq, Seq)
        final_attn_dist = avg_attn[0, -1, :] # Attention from Last Token to All Previous
        
        # 1. BOS Handling: Zero out index 0 (BOS)
        final_attn_dist[0] = 0.0
        
        # 2. Define Regions
        # Recent (Query equivalent): Last 256 tokens
        # Past (Ref equivalent): Index 1 to -(256+1)
        
        recent_start_idx = seq_len - RECENT_WINDOW
        
        attn_recent = final_attn_dist[recent_start_idx:]
        attn_past = final_attn_dist[1:recent_start_idx]
        
        sum_recent = attn_recent.sum().item()
        sum_past = attn_past.sum().item()
        total_attn = sum_recent + sum_past + 1e-9
        
        # 3. Region-Level Ratios (For Student Query-Attention Alignment)
        ratio_ref = sum_past / total_attn
        ratio_query = sum_recent / total_attn
        
        # 4. Trigger Condition: Did we need the past?
        if sum_past > sum_recent:
            # Significant Retrieval Case
            
            # 5. Chunk-Level Labels (For HyperNet)
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
                
                # Decode Text
                c_tokens = past_tokens[start:end]
                decoded = tokenizer.decode(c_tokens, skip_special_tokens=True)
                if len(decoded.strip()) == 0: 
                    # If empty chunk (just special tokens?), skip sample
                    valid_chunks = False
                    break
                chunk_texts.append(decoded)
            
            if not valid_chunks: continue

            # Normalize Chunk Scores
            total_chunk_score = sum(chunk_scores) + 1e-9
            chunk_probs = [s / total_chunk_score for s in chunk_scores]
            
            # Select Best Ref
            best_chunk_idx = np.argmax(chunk_probs)
            best_ref_text = chunk_texts[best_chunk_idx]
            
            # Recent Text (Query)
            query_text = tokenizer.decode(input_ids[0, recent_start_idx:], skip_special_tokens=True)
            
            sample_entry = {
                "query_text": query_text,       # The "User: Query" part
                "top_ref_text": best_ref_text,  # The "Ref" part
                "label_chunk_probs": chunk_probs, # HyperNet Target
                "label_attn_ratio_ref": ratio_ref, # Student Balance Target
                "label_attn_ratio_query": ratio_query # Student Balance Target
            }
            
            data_buffer.append(sample_entry)
            
            if len(data_buffer) % 50 == 0:
                print(f"Collected {len(data_buffer)} samples...")

    # Save
    print(f"Saving {len(data_buffer)} samples to {DATA_OUTPUT}")
    with open(DATA_OUTPUT, "w") as f:
        for entry in data_buffer:
            f.write(json.dumps(entry) + "\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=5000)
    args = parser.parse_args()
    NUM_SAMPLES = args.num_samples
    main()
