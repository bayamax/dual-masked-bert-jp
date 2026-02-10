
import json
import random
import os
from tqdm import tqdm
from transformers import AutoTokenizer

# Config
OUTPUT_FILE = "phase10_composite.jsonl"
RATIO_GENERAL = 0.5
RATIO_MATH = 0.3
RATIO_RETRIEVAL = 0.2

# File Paths (Assuming standard locations)
FILE_GENERAL = "phase8_alpaca_reasoning.jsonl"
FILE_MATH = "phase9_reasoning_gsm8k.jsonl"
FILE_RETRIEVAL = "phase7_correction_data.jsonl" # Or verify_phase7_unified logs if not available

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

def load_jsonl(filename):
    data = []
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found. Skipping.")
        return []
    with open(filename, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                continue
    print(f"Loaded {len(data)} from {filename}")
    return data

def format_general(item):
    # Phase 8 Format: USER: ... ASSISTANT: ...
    # Already processed? Check keys.
    # Expected keys: "text", "z_vector"
    return item

def format_math(item):
    # Phase 9 Format: USER: Solve... <think>...
    return item

def main():
    print("Loading datasets...")
    general_data = load_jsonl(FILE_GENERAL)
    math_data = load_jsonl(FILE_MATH)
    retrieval_data = load_jsonl(FILE_RETRIEVAL)
    
    # Calculate target counts based on the smallest available set relative to its ratio?
    # Or just use max available and downsample logic.
    
    # Let's aim for ~10k total for a solid epoch.
    TOTAL_TARGET = 10000
    
    n_general = int(TOTAL_TARGET * RATIO_GENERAL)
    n_math = int(TOTAL_TARGET * RATIO_MATH)
    n_retrieval = int(TOTAL_TARGET * RATIO_RETRIEVAL)
    
    # Resample
    composite = []
    
    # General
    if general_data:
        composite.extend(random.choices(general_data, k=n_general))
        
    # Math
    if math_data:
        composite.extend(random.choices(math_data, k=n_math))
        
    # Retrieval
    if retrieval_data:
        composite.extend(random.choices(retrieval_data, k=n_retrieval))
    else:
        # If retrieval data is missing, redistribute to General
        print("Retrieval data missing, adding more General data.")
        composite.extend(random.choices(general_data, k=n_retrieval))
        
    random.shuffle(composite)
    
    print(f"Composite Dataset created: {len(composite)} samples.")
    print(f"  General: {n_general}")
    print(f"  Math: {n_math}")
    print(f"  Retrieval: {n_retrieval if retrieval_data else 0}")
    
    with open(OUTPUT_FILE, 'w') as f:
        for item in composite:
            f.write(json.dumps(item) + "\n")
            
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
