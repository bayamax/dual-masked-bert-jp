"""
Phase 7 SFT Verification Script

Tests the SFT-trained model to see if it properly reads [参照情報: ...] blocks.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
import os

# Use SFT-trained adapter
MODEL_PATH = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
SFT_ADAPTER = "phase7_sft_lora_epoch2"
RAW_DATA = "phase7_raw_cot.jsonl"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model with SFT adapter
    print("Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    print(f"Loading SFT adapter from {SFT_ADAPTER}...")
    model = PeftModel.from_pretrained(base_model, SFT_ADAPTER)
    model.eval()
    
    # Load test sample
    print("Loading test sample...")
    with open(RAW_DATA, 'r') as f:
        sample = json.loads(f.readline())
    
    question = sample['question']
    cot = sample['generated_cot'][:400]  # Truncate for context window
    
    # Format input like training
    input_text = f"[参照情報: {cot}]\n\nUser: {question}\nModel:"
    
    print("-" * 50)
    print(f"Question: {question}")
    print("-" * 50)
    
    # Generate
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=50,
            do_sample=False,  # Greedy for consistency
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Decode only new tokens
    new_tokens = outputs[0][inputs.input_ids.shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    print("Model Response:")
    print(response)
    print("-" * 50)
    
    # Test a few more samples
    print("\n=== Additional Tests ===\n")
    
    with open(RAW_DATA, 'r') as f:
        lines = f.readlines()[:5]
    
    for i, line in enumerate(lines[1:], start=2):  # Skip first (already tested)
        sample = json.loads(line)
        question = sample['question']
        cot = sample['generated_cot'][:400]
        
        input_text = f"[参照情報: {cot}]\n\nUser: {question}\nModel:"
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
        
        print(f"[Test {i}] Q: {question[:80]}...")
        print(f"[Test {i}] A: {response}")
        print()

if __name__ == "__main__":
    main()
