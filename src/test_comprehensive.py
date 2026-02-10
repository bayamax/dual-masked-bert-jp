
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
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

class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_ids):
        self.stop_ids = stop_ids
    def __call__(self, input_ids, scores, **kwargs):
        for stop_id in self.stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

def load_system(device, lora_path):
    print(f"Loading Base: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, device_map=device)
    
    print(f"Loading LoRA: {lora_path}")
    model = PeftModel.from_pretrained(base, lora_path)
    model.eval()
    
    hypernet_path = os.path.join(lora_path, "hypernet.pt")
    if os.path.exists(hypernet_path):
        print(f"Loading HyperNet: {hypernet_path}")
        hypernet = HyperNetHead().to(device)
        hypernet.load_state_dict(torch.load(hypernet_path, map_location=device))
        hypernet.eval()
    else:
        print("HyperNet not found (skipping).")
        hypernet = None
        
    return model, hypernet, tokenizer

def generate(model, tokenizer, prompt, device, stop_words=None):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    
    stop_ids = [tokenizer.eos_token_id]
    
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=300,
            do_sample=True,
            temperature=0.7,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=stop_ids
        )
    generated = tokenizer.decode(out[0], skip_special_tokens=True)
    
    # Manual Stop String Truncation
    if stop_words:
        for sw in stop_words:
            if sw in generated[len(prompt):]:
                generated = generated[:generated.find(sw, len(prompt))]
                
    return generated

def run_tests(model, tokenizer, device, test_cases, model_name):
    print(f"\n=== Running Tests for {model_name} ===")
    
    for i, test in enumerate(test_cases):
        print(f"\n[Test {i+1}: {test['type']}]")
        print(f"Input: {test['prompt']}")
        
        # Format Prompt (Using Standard Chat Template structure we trained with)
        # Phase 9 trained with: USER: <Instruction>\nQuestion:\n<Q>\n\nAnswer: ASSISTANT:
        # Phase 8 trained with: USER: <Instruction>\nASSISTANT:
        
        if test['type'] == 'math':
            # GSM8K Format
            final_prompt = f"{USER_TAG}Solve the following math problem step-by-step.\nEnclose your thinking process in <think> tags.\nFinally, provide the answer.\n\nQuestion:\n{test['prompt']}\n\nAnswer:{USER_END}{MODEL_TAG}"
        else:
            # General Format
            final_prompt = f"{USER_TAG}{test['prompt']}{USER_END}{MODEL_TAG}"
            
        output = generate(model, tokenizer, final_prompt, device, stop_words=["USER:", "Question:", "ASSISTANT:"])
        
        # Extract just the response part
        response = output[len(final_prompt):].strip()
        print(f"Output:\n{response}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase9_path", type=str, default="phase9_math_model_epoch2")
    parser.add_argument("--phase8_path", type=str, default="phase8_combined_model_debug_epoch0")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Define Test Cases
    cases_math = [
        {"type": "math", "prompt": "What is 25 * 4 + 10?"},
        {"type": "math", "prompt": "Janet has 3 cats. She buys 2 more. How many legs do they have in total?"},
        {"type": "math", "prompt": "If I have 10 apples and eat 2, then buy 5 more, how many do I have?"}
    ]
    
    cases_general = [
        {"type": "general", "prompt": "Which is heavier, a kilogram of steel or a kilogram of feathers?"},
        {"type": "general", "prompt": "Write a short haiku about a robot."},
        {"type": "general", "prompt": "List 3 benefits of regular exercise."}
    ]
    
    # 1. Test Phase 9 (Math Model) on Math Tasks
    try:
        model9, _, tokenizer9 = load_system(device, args.phase9_path)
        run_tests(model9, tokenizer9, device, cases_math, "Phase 9 (Math)")
        del model9
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Failed to test Phase 9: {e}")

    # 2. Test Phase 8 (General Model) on General Tasks
    try:
        model8, _, tokenizer8 = load_system(device, args.phase8_path)
        run_tests(model8, tokenizer8, device, cases_general, "Phase 8 (General)")
        del model8
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Failed to test Phase 8: {e}")

if __name__ == "__main__":
    main()
