"""
Phase 8 Inference & Verification Script

Objectives:
1. Verify the model generates <think>...</think> before retrieval.
2. Verify the model outputs a valid Z-vector (<ref>...</ref>).
3. Test "Retry Behavior":
   - Force-feed a "Wrong Chunk" (simulated failure).
   - Check if the model re-generates a new Z-vector properly.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import argparse
import os

# Config
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_PATH = "phase8_combined_model_epoch0" # Adjust as needed
HYPERNET_PATH = "phase8_combined_model_epoch0/hypernet.pt"

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

def generate_step(model, tokenizer, input_ids, device, max_new_tokens=100):
    # Standard generation until a special token or length
    # Here we just generate text.
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    return out

def get_z_vector(model, hypernet, input_ids):
    # Get the hidden state of the last token
    with torch.no_grad():
        out = model.get_base_model()(input_ids=input_ids, output_hidden_states=True)
        last_hidden = out.hidden_states[-1][:, -1, :] # [B, H]
        z = hypernet(last_hidden.float()) # [B, Z]
        z = F.normalize(z, p=2, dim=1)
    return z

def verify_retry(model, hypernet, tokenizer, device):
    prompt = f"{USER_TAG}What is the capital of France?{USER_END}{MODEL_TAG}"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    print(f"\nrunning with Input: {prompt}")
    
    # 1. Generate Think + Initial Retrieval
    # We expect: <think> ... </think> <ref>
    # Since we can't easily stop at <ref>, we generate a bit.
    
    output_ids = generate_step(model, tokenizer, input_ids, device, max_new_tokens=100)
    output_text = tokenizer.decode(output_ids[0])
    print(f"Generated (Initial): {output_text}")
    
    # Check for <think>
    if "<think>" in output_text:
        print("✅ <think> tag found.")
    else:
        print("❌ <think> tag MISSING.")

    # 2. Simulate "Wrong Retrieval"
    # We verify if we can force the model to "Retry".
    # We construct a sequence: Prompt -> <think>... -> <ref>WRONG_CHUNK</ref> -> [TARGET]
    # The model should predict a NEW Z at [TARGET].
    
    # Let's craft a manual context
    wrong_chunk = "<ref>The capital of Mars is Elon City.</ref>"
    
    # Assume the model generated up to the end of <think>
    # If it didn't generate <think>, force it using a preset.
    
    preset_think = "<think>I need to find the capital of France. I will retrieve it.</think>"
    context_ids = tokenizer.encode(f"{prompt}{preset_think}{wrong_chunk}", return_tensors="pt").to(device)
    
    print("Injecting Wrong Chunk...")
    
    # Get Z from this state
    z_retry = get_z_vector(model, hypernet, context_ids)
    
    print(f"Retry Z Vector Norm: {z_retry.norm().item():.4f}")
    
    # Ideally we compare this Z with the Z from the "Clean" prompt.
    # Get Z from clean prompt (approximate, since we need to be at the exact generation step for Z)
    # Actually, the model outputs Z at the token BEFORE <ref>.
    # So we need to feed `Prompt + <think>...` and get Z.
    
    pre_ref_ids = tokenizer.encode(f"{prompt}{preset_think}", return_tensors="pt").to(device)
    z_initial = get_z_vector(model, hypernet, pre_ref_ids)
    
    # Similarity
    sim = F.cosine_similarity(z_initial, z_retry).item()
    print(f"Similarity between Initial Z and Retry Z: {sim:.4f}")
    
    if sim > 0.95:
        print("⚠️ Warning: Retry Z is very similar to Initial Z. Model might be ignoring the wrong chunk.")
    else:
        print("✅ Retry Z is distinct. The model reacted to the wrong chunk!")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_path", type=str, default="phase8_combined_model_epoch0")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, hypernet, tokenizer = load_model(device, args.lora_path)
    
    verify_retry(model, hypernet, tokenizer, device)

if __name__ == "__main__":
    main()
