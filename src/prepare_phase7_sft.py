
import json
import random
import os
from tqdm import tqdm
from datasets import load_dataset
import spacy

# Ensure spacy model is present or use simple split
# Assuming we don't have spacy on remote, we use simple splitting.

def simple_sent_tokenize(text):
    return text.replace('\n', ' ').split('. ')

def main():
    print("Generating SFT Extraction Data...")
    
    # Load FineWeb-Edu (Sample)
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    
    output_file = "sft_extraction_50k.jsonl"
    target_count = 50000
    current_count = 0
    
    with open(output_file, 'w') as f:
        pbar = tqdm(total=target_count)
        
        for item in ds:
            text = item['text']
            
            # Simple heuristics to get good chunks
            if len(text) < 2000: continue
            
            # Parse sentences (rough)
            sentences = simple_sent_tokenize(text)
            if len(sentences) < 10: continue
            
            # Create triplet
            # 1. Pick a target sentence (FACT)
            # Avoid first/last to ensure context
            idx = random.randint(2, len(sentences) - 3)
            fact = sentences[idx].strip()
            
            # Check quality
            if len(fact.split()) < 10: continue # Too short
            if len(fact) > 300: continue # Too long
            
            # 2. Formulate Context
            # Context includes the fact (simulating "Retrieved Chunk")
            # We take a window around it
            start = max(0, idx - 3)
            end = min(len(sentences), idx + 4)
            context_sents = sentences[start:end]
            context = ". ".join(context_sents)
            
            # 3. Formulate Prompt
            # "Extract the sentence that discusses [Topic]..." is hard to automate without Topic.
            # "Complete the sentence starting with..."
            
            prefix = " ".join(fact.split()[:4]) # First 4 words
            instruction = f"Based on the following retrieved context, complete the sentence starting with '{prefix}'."
            
            # Input format: [Instruction] + [Context]
            # Actually our pipeline is [Target] -> [Context] -> [Current]
            # Wait, Phase 6 Training Pipeline is:
            # Pass 1: Target (Fact) -> KV
            # Pass 2: Current (Query) + KV -> Answer
            
            # For SFT, we want the MODEL to Generate Answer.
            # So Input = Instruction + Prefix.
            # Context = The Retrieved Chunk (Injected via Hippocampus/KV).
            
            # But here we are creating JSONL for Training.
            # Format: {"context": "...", "instruction": "...", "response": "..."}
            
            entry = {
                "context": context, # This will be "Injected" (Z-vector or KV)
                "instruction": instruction, # This is the "Current Input"
                "response": fact # This is the "Target Generation"
            }
            
            f.write(json.dumps(entry) + "\n")
            current_count += 1
            pbar.update(1)
            
            if current_count >= target_count:
                break
                
    print(f"Saved {current_count} examples to {output_file}")

if __name__ == "__main__":
    main()
