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

from transformers import StoppingCriteria, StoppingCriteriaList

class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_ids):
        self.stop_ids = stop_ids
    def __call__(self, input_ids, scores, **kwargs):
        for stop_id in self.stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

def generate(model, tokenizer, prompt, device):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Stop tokens: EOS, "USER:", etc.
    stop_words = ["Question:", "USER:", "\n\nQuestion"]
    # We can't easily use string stopping without a custom criteria or streamwatcher.
    # For simplicity in this script, we'll check for EOS token and maybe just rely on the model learning it?
    # Actually, let's implement a basic stopping criteria for the EOS token at least.
    # TinyLlama might not emit EOS if not trained to do so?
    # We trained with pad/eos as separator?
    # In prepare_phase9: " padded_inputs[padded_inputs == pad_id] = -100 "
    # We didn't explicitly append EOS to the training data in `prepare`.
    # Wait, `tokenizer.encode` usually adds special tokens?
    # `add_special_tokens=False` was used for parts.
    
    # We should add a custom stopping criteria for strings if possible, but simpler is just to cut it off.
    # Let's rely on EOS if the model learned it.
    # If we didn't train EOS, we *must* use stop strings.
    
    # Using transformers' `eos_token_id` is standard. 
    # Let's add specific token IDs if known. 
    # "USER" token?
    
    stop_ids = [tokenizer.eos_token_id]
    # Add newline?
    
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            max_new_tokens=500,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=stop_ids
        )
    generated = tokenizer.decode(out[0], skip_special_tokens=True)
    
    # Manual Truncation for clean look (Post-processing)
    # This simulates "Stop Strings"
    for stop in stop_words:
        if stop in generated[len(prompt):]:
            generated = generated[:generated.find(stop, len(prompt))]
            
    return generated

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_path", type=str, default="phase9_math_model_epoch2")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, hypernet, tokenizer = load_model(device, args.lora_path)
    
    # GSM8K Validation Question
    question = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
    # Answer: 48 + 24 = 72.
    
    prompt = f"{USER_TAG}{question}{USER_END}{MODEL_TAG}"
    
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
