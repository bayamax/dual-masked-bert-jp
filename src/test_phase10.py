
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os

# Config
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_PATH = "phase10_composite_model"
USER_TAG = "USER: "
MODEL_TAG = "ASSISTANT: "
USER_END = "\n"

def load_model():
    print(f"Loading Base: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    
    print(f"Loading Phase 10 Adapter: {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    model.eval()
    return model, tokenizer

def generate(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=True,
            temperature=temp,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)

def run_test(model, tokenizer, prompt, task_type, temp=0.6):
    print(f"\n--- Test: {task_type} (Temp={temp}) ---")
    print(f"In: {prompt}")
    
    # Format
    if task_type == "math":
        full_prompt = f"{USER_TAG}Solve this step-by-step.\nQuestion:\n{prompt}\n\nAnswer:{USER_END}{MODEL_TAG}"
    else:
        full_prompt = f"{USER_TAG}{prompt}{USER_END}{MODEL_TAG}"
        
    output = generate(model, tokenizer, full_prompt, temp)
    response = output[len(full_prompt):].strip()
    print(f"Out:\n{response}\n")

def main():
    model, tokenizer = load_model()
    
    # 1. Math Reasoning - Test Strictness
    prompt = "If I have 7 apples and eat 2, then buy 5 more, how many do I have?"
    
    # Default (0.6)
    run_test(model, tokenizer, prompt, "math", temp=0.6)
    
    # Strict (0.2) - Expected to be less creative/hallucinatory
    run_test(model, tokenizer, prompt, "math", temp=0.2)
    
    # Very Strict (0.1)
    run_test(model, tokenizer, prompt, "math", temp=0.1)

if __name__ == "__main__":
    main()
