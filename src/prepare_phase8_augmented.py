"""
Phase 8 Data Prep: Argumented Reasoning Alpaca with Correction Training

Objective:
1. Load Phase 8 Reasoning Alpaca (JSONL).
2. Chunk the content (Think + Answer).
3. Compute Z-vectors using the Phase 7.6 Corrected Model.
4. Augment with "Mistake Injection":
   - 30% of the time, insert a Wrong Z before the Correct Z.
   - Sequence: [Context] -> [Wrong Z] -> [Correct Z] -> [Next Text]
   - Mask loss on Wrong Z? No, it's input. We predict Correct Z.

Output: phase8_augmented_chunks.pt
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
import random
import os
from tqdm import tqdm

# Config
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_PATH = "phase7_online_correction_epoch2"
HYPERNET_PATH = "phase7_online_correction_epoch2/hypernet.pt"
INPUT_JSONL = "phase8_reasoning_alpaca.jsonl"
OUTPUT_PT = "phase8_augmented_dataset.pt" # Contains tokenized sequences for training

CHUNK_SIZE = 127
AUGMENT_PROB = 0.3
MAX_SEQ_LEN = 2048

# Special Tokens
USER_TAG = "<user>"
USER_END = "</user>\n"
MODEL_TAG = "<model>"
THINK_TAG = "<think>\n"
THINK_END = "</think>\n"
REF_TAG = "<ref>" # We might use special tokens for Z placeholders if doing token-based training
# But our current architecture uses Z-vector injection at specific points.

# Wait, how do we train "Z input"?
# In Phase 7 Unified, we interleaved text and Z.
# We need to format the data as a list of "Segments".
# Segment: {type: "text", ids: [...] } or {type: "z", vector: [...], target: True/False}

class HyperNetHead(nn.Module):
    def __init__(self, input_dim=2048, output_dim=2048):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.proj(x)

def load_jsonl(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load Model for Z computation
    print("Loading Model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, device_map=device)
    model = PeftModel.from_pretrained(base, LORA_PATH)
    model.eval()
    
    hypernet = HyperNetHead().to(device)
    hypernet.load_state_dict(torch.load(HYPERNET_PATH, map_location=device))
    hypernet.eval()
    
    # Load Data
    raw_data = load_jsonl(INPUT_JSONL)
    print(f"Loaded {len(raw_data)} samples.")
    
    # Also load retrieval corpus to mine "Random Negatives" (or Hard ones?)
    # For "Re-draw" training, random wrong Z is sufficient to teach "If match is bad, try again".
    # But Hard Negative is better.
    # Let's verify compatibility with phase7_cot_chunks.pt (for index).
    # Ideally we use the SAME index throughout.
    # But here we just need *some* wrong Z vectors.
    # We can perform Hard Negative Mining on the fly? No, slow.
    # We can specific "Random Z" from the batch or random projection.
    # Let's generate random unit vectors for now? No, model might distinguish "Noise" vs "Real Z".
    # We must use Real Zs from other samples.
    
    augmented_samples = []
    
    # We will process in batches to compute Z and build dataset
    batch_size = 8 
    
    for i in tqdm(range(0, len(raw_data), batch_size)):
        batch = raw_data[i : i+batch_size]
        
        # 1. Parse content
        # Alpaca JSONL: {instruction, input, output} (Output has <think>)
        
        texts_to_embed = []
        map_text_to_sample = [] # (batch_idx, chunk_idx)
        
        processed_batch = []
        
        for b_i, item in enumerate(batch):
            instruction = item.get('instruction', '')
            inp = item.get('input', '')
            output = item.get('output', '') # Contains <think>...</think> Answer...
            
            prompt = f"{USER_TAG}{instruction}\n{inp}{USER_END}{MODEL_TAG}\n"
            
            # Split output into chunks (127 tokens)
            # We want to insert Z *between* chunks.
            # Tokenize output
            out_ids = tokenizer.encode(output, add_special_tokens=False)
            
            chunks = []
            for j in range(0, len(out_ids), CHUNK_SIZE):
                chunk_ids = out_ids[j : j+CHUNK_SIZE]
                chunks.append(chunk_ids)
            
            processed_batch.append({
                'prompt_ids': tokenizer.encode(prompt, add_special_tokens=False),
                'chunk_ids_list': chunks,
                'z_vectors': [None] * len(chunks) # To be filled
            })
            
            # Prepare for Z computation
            # For each chunk, we need to forward pass to get Z.
            # But wait, Z is computed from the *End* of the chunk?
            # Or is it computed from the *History*?
            # In Phase 7, we trained: Input -> Z. Z represents the *next* content?
            # No, Z represents the *current* chunk to be retrieved.
            # Retrieval Query is generated *before* the chunk.
            # So: [Prompt] -> [Generate Query Z] -> [Retrieve Chunk] -> [Generate Chunk Tokens]
            # Training Target for Query Z is the Z-vector of the Chunk.
            # So we need Z-vector of the Chunk text itself.
            
            for c_i, c_ids in enumerate(chunks):
                # Chunk Text
                text = tokenizer.decode(c_ids)
                texts_to_embed.append(text)
                map_text_to_sample.append((b_i, c_i))
                
        # 2. Compute Z-vectors for all chunks in batch
        # This aligns the target Z with the current model
        if texts_to_embed:
            # Batch Encode
             with torch.no_grad():
                # Sub-batching for embedding
                emb_bs = 16
                for j in range(0, len(texts_to_embed), emb_bs):
                    sub_texts = texts_to_embed[j : j+emb_bs]
                    inputs = tokenizer(sub_texts, return_tensors="pt", padding=True, truncation=True, max_length=CHUNK_SIZE).to(device)
                    
                    # Attention Mask Sum hack for last token
                    seq_lens = inputs.attention_mask.sum(dim=1) - 1
                    out = model.get_base_model()(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, output_hidden_states=True)
                    last_h = out.hidden_states[-1]
                    
                    batch_h = last_h[torch.arange(len(sub_texts), device=device), seq_lens, :]
                    batch_z = hypernet(batch_h.float())
                    batch_z = F.normalize(batch_z, p=2, dim=1)
                    
                    # Store back
                    for k, z in enumerate(batch_z):
                        global_idx = j + k
                        b_i, c_i = map_text_to_sample[global_idx]
                        processed_batch[b_i]['z_vectors'][c_i] = z.cpu()
                        
        # 3. Construct Training Sequences (Augmentation)
        # We need a pool of "Wrong Zs". We can use Zs from other samples in the batch.
        
        all_z_in_batch = []
        for p in processed_batch:
            all_z_in_batch.extend([z for z in p['z_vectors'] if z is not None])
            
        for p in processed_batch:
            prompt_ids = p['prompt_ids']
            chunks = p['chunk_ids_list']
            zs = p['z_vectors']
            
            # Format:
            # [Prompt]
            # [Ref Z0] [Chunk 0]
            # [Ref Z1] [Chunk 1] ...
            
            # Augmented:
            # [Prompt]
            # [Ref Wrong] [Ref Z0] [Chunk 0] ...
            
            final_sequence_segments = [] # List of {'type': 'text/z/wrong_z', 'data': ...}
            
            final_sequence_segments.append({'type': 'text', 'ids': prompt_ids})
            
            for c_ids, z in zip(chunks, zs):
                # Augment?
                if random.random() < AUGMENT_PROB and len(all_z_in_batch) > 1:
                    # Pick a wrong Z
                    while True:
                        wrong_z = random.choice(all_z_in_batch)
                        # Check cosine sim? If too close, pick another.
                        # For speed, just assume random choice is likely wrong.
                        if not torch.equal(wrong_z, z):
                            break
                            
                    final_sequence_segments.append({'type': 'z_input', 'vector': wrong_z, 'is_correct': False})
                    # After wrong Z, we MUST predict Correct Z.
                    # The "Correction" action is implicit in the sequence: WrongZ -> CorrectZ.
                    
                # Correct Z
                final_sequence_segments.append({'type': 'z_target', 'vector': z, 'is_correct': True})
                
                # Chunk Tokens
                final_sequence_segments.append({'type': 'text', 'ids': c_ids})
            
            augmented_samples.append(final_sequence_segments)
            
    print(f"Saving {len(augmented_samples)} augmented samples...")
    torch.save(augmented_samples, OUTPUT_PT)
    print("Done.")

if __name__ == "__main__":
    main()
