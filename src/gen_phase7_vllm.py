from vllm import LLM, SamplingParams
from datasets import load_dataset
import json
import argparse
import os

# Configuration
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
OUTPUT_FILE = "phase7_raw_cot.jsonl"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=5000)
    args = parser.parse_args()
    
    print(f"Loading vLLM Model: {MODEL_NAME}...")
    # Initialize vLLM
    # Note: vLLM usually handles memory management better than HF
    # We might need to adjust gpu_memory_utilization if OOMs occur, but default 0.9 is usually fine or high
    # Update: Lowering to 0.6 and enabling eager mode for stability on remote instance
    llm = LLM(model=MODEL_NAME, tensor_parallel_size=1, dtype="bfloat16", gpu_memory_utilization=0.6, enforce_eager=True) 
    
    print("Loading Dataset (GSM8K)...")
    dataset = load_dataset("gsm8k", "main", split="train")
    
    # Prepare Prompts
    prompts = []
    questions = []
    
    GSM_PROMPT = "Question: {question}\nAnswer:"
    
    count = 0
    for sample in dataset:
        if count >= args.num_samples: break
        prompts.append(GSM_PROMPT.format(question=sample['question']))
        questions.append(sample['question'])
        count += 1
        
    print(f"Generating {len(prompts)} samples...")
    
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.9,
        max_tokens=1024,
        stop=["Question:", "User:"] # Optional stop tokens
    )
    
    outputs = llm.generate(prompts, sampling_params)
    
    # Save Results
    print(f"Saving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w") as f:
        for i, output in enumerate(outputs):
            generated_text = output.outputs[0].text
            entry = {
                "question": questions[i],
                "prompt": prompts[i], # Good to save the prompt context
                "generated_cot": generated_text
            }
            f.write(json.dumps(entry) + "\n")

if __name__ == "__main__":
    main()
