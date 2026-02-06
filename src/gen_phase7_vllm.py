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
    # We restrict max_model_len to 4096 to save KV cache memory (default is 128k!)
    llm = LLM(model=MODEL_NAME, tensor_parallel_size=1, dtype="bfloat16", gpu_memory_utilization=0.9, enforce_eager=True, max_model_len=4096) 
    
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
        
    print(f"Generating {len(prompts)} samples in batches...")
    
    # Resume Logic
    start_index = 0
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r") as f:
            start_index = sum(1 for line in f)
        print(f"Found existing data. Resuming from sample {start_index}...")

    # Calculate remaining work
    prompts = prompts[start_index:]
    questions = questions[start_index:]
    
    if len(prompts) == 0:
        print("All samples already generated!")
        return

    batch_size = 500
    total_prompts = len(prompts)
    
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.9,
        max_tokens=1024,
        stop=["Question:", "User:"] 
    )

    for i in range(0, total_prompts, batch_size):
        batch_prompts = prompts[i : i + batch_size]
        batch_questions = questions[i : i + batch_size]
        
        print(f"Processing batch {i} to {i+batch_size}...")
        outputs = llm.generate(batch_prompts, sampling_params)
        
        # Append batch results
        with open(OUTPUT_FILE, "a") as f:
            for j, output in enumerate(outputs):
                generated_text = output.outputs[0].text
                entry = {
                    "question": batch_questions[j],
                    "prompt": batch_prompts[j], 
                    "generated_cot": generated_text
                }
                f.write(json.dumps(entry) + "\n")
                f.flush()
        
        # Optional: Force garbage collection if vLLM leaks memory heavily (usually not needed but safe)
        # import gc; gc.collect()
    
    print(f"Generation Complete. Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
