
import json
from datasets import load_dataset
from tqdm import tqdm

def main():
    print("Loading Wikitext-103 (Scaling to 50k)...")
    ds = load_dataset("wikitext", "wikitext-103-v1", split="train", streaming=False)
    
    output_file = "wiki_long_50k.jsonl"
    count = 0
    target = 50000 # Increased to 50k
    
    with open(output_file, 'w') as f:
        for item in tqdm(ds):
            text = item['text'].strip()
            # Filter for decent length to ensure good recursive logic
            if len(text) > 500 and not text.startswith('='):
                 entry = {"text": text}
                 f.write(json.dumps(entry) + "\n")
                 count += 1
            
            if count >= target:
                break
    
    print(f"Saved {count} articles to {output_file}")

if __name__ == "__main__":
    main()
