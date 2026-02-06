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
MAX_LENGTH = 2048
RECENT_WINDOW = 256 # Considered as "Query/Prompt" area
CHUNK_SIZE = 127 # set to 127 to allow 1 token for 'z' (Recursive Structure: 127 text + 1 z = 128)
NUM_SAMPLES = 5000

def main():
    print(f"Loading Teacher: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        torch_dtype=torch.bfloat16, 
        load_in_4bit=True,
        device_map="auto", 
        attn_implementation="eager" 
    )
    
    # Check context window
    print(f"Model File: {model.config._name_or_path}")
    
    # Configuration
    GSM_PROMPT = "Question: {question}\nAnswer:"
    
    print("Loading Dataset (GSM8K)...")
    dataset = load_dataset("gsm8k", "main", split="train")
    
    data_buffer = []
    
    # We iterate through GSM8K questions
    print(f"Starting Generation & Distillation on {NUM_SAMPLES} samples...")
    
    for i, sample in enumerate(tqdm(dataset)):
        if len(data_buffer) >= NUM_SAMPLES: break
        
        question = sample['question']
        prompt = GSM_PROMPT.format(question=question)
        
        # 1. Generate CoT (Teacher)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # We generate up to 1024 tokens of CoT
        with torch.no_grad():
            gen_ids = model.generate(
                inputs.input_ids, 
                max_new_tokens=1024, 
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.6,
                top_p=0.9
            )
        
        # The full text is what we analyze
        # (Prompt + Generated CoT)
        # Note: gen_ids includes prompt if we use standard generate, usually? 
        # Yes, model.generate returns full sequence by default or just new? 
        # It usually returns Input + New.
        
        tokens = gen_ids[0]
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        
        # 2. Analyze the Generated Text for Attention Loopbacks
        # We reuse the logic: "Did the end (recent) look at the beginning (ref)?"
        # We need to run a forward pass on this FULL sequence to get the attention map.
        
        # Truncate if too long (though standard GSM8K shouldn't exceed 2048)
        if len(tokens) > MAX_LENGTH:
             tokens = tokens[:MAX_LENGTH]
        
        input_ids = tokens.unsqueeze(0)
        seq_len = input_ids.shape[1]
        
        # Skip if too short to represent "Past + Recent"
        if seq_len < RECENT_WINDOW + CHUNK_SIZE: 
            del gen_ids, inputs
            torch.cuda.empty_cache()
            continue

        # Run Forward Pass to get Attentions
        with torch.no_grad():
            outputs = model(input_ids, output_attentions=True)
        
        # Extract Attention (Same logic as before)
        attentions = outputs.attentions[-4:] 
        avg_attn = torch.stack(attentions).mean(dim=(0, 2)) # (1, Seq, Seq)
        final_attn_dist = avg_attn[0, -1, :] 
        
        final_attn_dist[0] = 0.0 # Ignore BOS
        
        recent_start_idx = seq_len - RECENT_WINDOW
        
        attn_recent = final_attn_dist[recent_start_idx:]
        attn_past = final_attn_dist[1:recent_start_idx]
        
        sum_recent = attn_recent.sum().item()
        sum_past = attn_past.sum().item()
        total_attn = sum_recent + sum_past + 1e-9
        
        ratio_ref = sum_past / total_attn
        ratio_query = sum_recent / total_attn
        
        # Trigger Condition
        if sum_past > sum_recent:
            # Found a loopback in the CoT!
            
            # Chunking Logic (Same as before)
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
                    valid_chunks = False; break
                chunk_texts.append(decoded)
            
            if valid_chunks:
                total_chunk_score = sum(chunk_scores) + 1e-9
                chunk_probs = [s / total_chunk_score for s in chunk_scores]
                best_ref_text = chunk_texts[np.argmax(chunk_probs)]
                
                # Query Text (The recent thought process)
                query_text = tokenizer.decode(input_ids[0, recent_start_idx:], skip_special_tokens=True)
                
                sample_entry = {
                    "query_text": query_text,
                    "top_ref_text": best_ref_text,
                    "label_chunk_probs": chunk_probs,
                    "label_attn_ratio_ref": ratio_ref,
                    "label_attn_ratio_query": ratio_query
                }
                data_buffer.append(sample_entry)
                
                if len(data_buffer) % 10 == 0:
                    print(f"Collected {len(data_buffer)} CoT samples...")
        
        # Cleanup
        del outputs, gen_ids, inputs
        torch.cuda.empty_cache()

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
