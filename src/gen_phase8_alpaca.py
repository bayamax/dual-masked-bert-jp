from vllm import LLM, SamplingParams
from datasets import load_dataset
import json
import argparse
import os

# Configuration
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
OUTPUT_FILE = "phase8_reasoning_alpaca.jsonl"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=52002) # Alpaca is ~52k
    args = parser.parse_args()
    
    print(f"Loading vLLM Model: {MODEL_NAME}...")
    # Initialize vLLM
    # Memory optimization: Limit max_model_len to prevent OOM with large batch contexts
    llm = LLM(
        model=MODEL_NAME, 
        tensor_parallel_size=1, 
        dtype="bfloat16", 
        gpu_memory_utilization=0.9, 
        enforce_eager=True, 
        max_model_len=4096
    ) 
    
    print("Loading Dataset (tatsu-lab/alpaca)...")
    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    
    # Prepare Prompts
    prompts = []
    raw_entries = [] # To keep metadata
    
    # Alpaca Prompt Template
    ALPACA_PROMPT = (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n"
        "### Input:\n{input}\n\n"
        "### Response:"
    )
    
    ALPACA_NO_INPUT_PROMPT = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n"
        "### Response:"
    )
    
    count = 0
    for sample in dataset:
        if count >= args.num_samples: break
        
        if sample.get('input', "").strip():
            prompt = ALPACA_PROMPT.format(instruction=sample['instruction'], input=sample['input'])
        else:
            prompt = ALPACA_NO_INPUT_PROMPT.format(instruction=sample['instruction'])
            
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
        stop=["### Instruction:", "### Response:"] 
    )

    for i in range(0, total_prompts, batch_size):
        batch_prompts = prompts[i : i + batch_size]
        batch_raw = raw_entries[i : i + batch_size]
        
        print(f"Processing batch {i} to {i+batch_size}...")
        outputs = llm.generate(batch_prompts, sampling_params)
        
        # Append batch results
        with open(OUTPUT_FILE, "a") as f:
            for j, output in enumerate(outputs):
                generated_text = output.outputs[0].text
                
                # Check if <think> exists
                has_think = "<think>" in generated_text
                
                entry = {
                    "instruction": batch_raw[j]['instruction'],
                    "input": batch_raw[j]['input'],
                    "original_output": batch_raw[j]['output'], # Keep original for reference
                    "generated_cot": generated_text,
                    "has_think": has_think
                }
                f.write(json.dumps(entry) + "\n")
                f.flush()
        
    print(f"Generation Complete. Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
