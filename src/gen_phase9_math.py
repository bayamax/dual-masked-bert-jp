from vllm import LLM, SamplingParams
from datasets import load_dataset
import json
import argparse
import os

# Configuration
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
OUTPUT_FILE = "phase9_reasoning_gsm8k.jsonl"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=10000) # GSM8K train is ~7.5k
    args = parser.parse_args()
    
    print(f"Loading vLLM Model: {MODEL_NAME}...")
    # Initialize vLLM
    # Memory optimization: Limit max_model_len to prevent OOM
    llm = LLM(
        model=MODEL_NAME, 
        tensor_parallel_size=1, 
        dtype="bfloat16", 
        gpu_memory_utilization=0.9, 
        enforce_eager=True, 
        max_model_len=4096
    ) 
    
    print("Loading Dataset (gsm8k)...")
    dataset = load_dataset("gsm8k", "main", split="train")
    
    # Prepare Prompts
    prompts = []
    raw_entries = [] # To keep metadata
    
    # Prompt Template
    MATH_PROMPT = (
        "Solve the following math problem step-by-step.\n"
        "Enclose your thinking process in <think> tags.\n"
        "Finally, provide the answer.\n\n"
        "Question:\n{question}\n\n"
        "Answer:"
    )
    
    count = 0
    for sample in dataset:
        if count >= args.num_samples: break
        
        prompt = MATH_PROMPT.format(question=sample['question'])
            
        prompts.append(prompt)
        raw_entries.append(sample)
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
    raw_entries = raw_entries[start_index:]
    
    if len(prompts) == 0:
        print("All samples already generated!")
        return

    batch_size = 500
    total_prompts = len(prompts)
    
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.9,
        max_tokens=2048, # Allow for long reasoning
        stop=["Question:", "Answer:"] # Stop if it tries to generate a new question
    )

    for i in range(0, total_prompts, batch_size):
        end_idx = min(i + batch_size, total_prompts)
        batch_prompts = prompts[i : end_idx]
        batch_raw = raw_entries[i : end_idx]
        
        print(f"Processing batch {i} to {end_idx}...")
        outputs = llm.generate(batch_prompts, sampling_params)
        
        # Append batch results
        with open(OUTPUT_FILE, "a") as f:
            for j, output in enumerate(outputs):
                generated_text = output.outputs[0].text
                
                # Check for <think>
                has_think = "<think>" in generated_text
                
                entry = {
                    "question": batch_raw[j]['question'],
                    "ground_truth_answer": batch_raw[j]['answer'],
                    "generated_cot": generated_text,
                    "has_think": has_think
                }
                f.write(json.dumps(entry) + "\n")
                f.flush()
        
    print(f"Generation Complete. Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
