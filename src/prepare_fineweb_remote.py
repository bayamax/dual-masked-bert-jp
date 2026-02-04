
import json
from datasets import load_dataset
from tqdm import tqdm
import os

def main():
    print("Loading FineWeb-Edu (Sample 10BT)...")
    # Streaming mode is ESSENTIAL for 10BT dataset to avoid downloading everything
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    
    output_file = "fineweb_edu_500k.jsonl"
    target = 500000 # 500k docs
    count = 0
    
    print(f"Target: {target} documents. Output: {output_file}")
    
    with open(output_file, 'w') as f:
        # Use simple iteration for streaming
        pbar = tqdm(total=target)
        
        for item in ds:
            text = item['text']
            # Filter for Quality & Length
            # FineWeb-Edu has 'score' field?
            # Sample-10BT usually has highly rated content.
            # We filter for length to ensure recursive unrolling is useful.
            
            if len(text) > 1000:
                # Store
                entry = {"text": text}
                f.write(json.dumps(entry) + "\n")
                count += 1
                pbar.update(1)
                
            if count >= target:
                break
                
    print(f"Saved {count} documents to {output_file}")

if __name__ == "__main__":
    main()
