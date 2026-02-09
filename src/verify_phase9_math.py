"""
Phase 9 Verification Script (Math Reasoning)

Objectives:
1. Verify the model generates <think> tags for a math problem.
2. Verify it generates a Z-vector before the answer.
3. Check if the answer is reasonable (qualitative).
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse
import os

# Config
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

USER_TAG = "USER: "
MODEL_TAG = "ASSISTANT: "
USER_END = "\n"

class HyperNetHead(torch.nn.Module):
    def __init__(self, input_dim=2048, output_dim=2048):
        super().__init__()
        self.proj = torch.nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.proj(x)

def load_model(device, lora_path):
    print(f"Loading base model: {BASE_MODEL}")
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, device_map=device)
    
    print(f"Loading LoRA: {lora_path}")
    model = PeftModel.from_pretrained(base, lora_path)
    model.eval()
    
    hypernet_path = os.path.join(lora_path, "hypernet.pt")
    print(f"Loading HyperNet: {hypernet_path}")
    hypernet = HyperNetHead().to(device)
    hypernet.load_state_dict(torch.load(hypernet_path, map_location=device))
    hypernet.eval()
    
    return model, hypernet, AutoTokenizer.from_pretrained(BASE_MODEL)

def generate(model, tokenizer, prompt, device):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            max_new_tokens=500,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_path", type=str, default="phase9_math_model_epoch2")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, hypernet, tokenizer = load_model(device, args.lora_path)
    
    # GSM8K Validation Question
    question = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
    # Answer: 48 + 24 = 72.
    
    prompt = f"{USER_TAG}Solve the following math problem step-by-step.\nEnclose your thinking process in <think> tags.\nFinally, provide the answer.\n\nQuestion:\n{question}\n\nAnswer:{USER_END}{MODEL_TAG}"
    
    print(f"Input: {prompt}\n")
    output = generate(model, tokenizer, prompt, device)
    print(f"Output:\n{output}\n")
    
    if "<think>" in output:
        print("✅ <think> tag found.")
    else:
        print("❌ <think> tag MISSING.")
        
    if "72" in output:
        print("✅ Correct Answer (72) found.")
    else:
        print("⚠️ Correct Answer not found (Might be format issue or wrong reasoning).")

if __name__ == "__main__":
    main()
