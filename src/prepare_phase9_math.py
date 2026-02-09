import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
import os
from tqdm import tqdm
import argparse

# Config
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# Use Phase 8 model for latest Z-space
LORA_PATH = "phase8_combined_model_debug_epoch0" 
HYPERNET_PATH = "phase8_combined_model_debug_epoch0/hypernet.pt"

INPUT_JSONL = "phase9_reasoning_gsm8k.jsonl"
OUTPUT_FILE = "phase9_math_dataset.pt"

CHUNK_SIZE = 127
USER_TAG = "USER: "
MODEL_TAG = "ASSISTANT: "
USER_END = "\n"

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print("Loading Model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, device_map=device)
    base.gradient_checkpointing_disable() # Inference mode
    
    print(f"Loading LoRA: {LORA_PATH}")
    model = PeftModel.from_pretrained(base, LORA_PATH)
    model.eval()
    
    hypernet = HyperNetHead().to(device)
    hypernet.load_state_dict(torch.load(HYPERNET_PATH, map_location=device))
    hypernet.eval()
    
    raw_data = load_jsonl(INPUT_JSONL)
    print(f"Loaded {len(raw_data)} samples.")
    
    if args.limit > 0:
        raw_data = raw_data[:args.limit]
    
    processed_samples = []
    batch_size = 32
    
    torch.backends.cuda.matmul.allow_tf32 = True
    
    for i in tqdm(range(0, len(raw_data), batch_size)):
        batch = raw_data[i : i+batch_size]
        
        texts_to_embed = []
        map_text_to_sample = [] # (batch_idx_in_loop)
        
        current_batch_data = [] # Stores {prompt_ids, think_ids, answer_ids, z_vector}
        
        for b_i, item in enumerate(batch):
            q = item['question']
            cot = item['generated_cot'] 
            # cot contains <think>...Answer
            
            # Parse CoT
            # Structure: <think>...</think>Answer...
            # We want to insert Z *after* <think> and *before* Answer? 
            # Or Z is just the representation of the Answer?
            # User requirement: "Do not update tokens". 
            # Standard Retrieval-Augmented Generation:
            # Query -> Retrieve -> Answer
            # Reasoning-Augmented:
            # Query -> Think -> Retrieve -> Answer
            
            # Let's assume we want to associate the Z with the *Answer*.
            # So the model Thinks, then Outputs Z (which retrieves context/verifies), then Answers.
            
            # We need to split COT into THINK and ANSWER parts if possible.
            # DeepSeek R1 output usually: "<think>...</think>\nAnswer"
            
            parts = cot.split("</think>")
            if len(parts) == 2:
                think_part = parts[0] + "</think>"
                answer_part = parts[1].strip()
            else:
                # Fallback: Treat whole thing as Answer? Or whole thing as Think?
                # If no tag, maybe it's just answer.
                think_part = ""
                answer_part = cot
            
            if len(answer_part) < 5:
                # Skip samples where answer is too short or missing
                continue

            # We want to embed the ANSWER to get its Z.
            # Wait, do we index the Answer?
            # Or do we train the model to predict the Z that *would* retrieve the Answer?
            # In Phase 7/8, we trained Z to match the content of the *next chunk*.
            # So here: Input = Q + Think. Target = Z_of_Answer. Next = Answer.
            
            # Construct Prompt
            prompt_str = f"{USER_TAG}{q}{USER_END}{MODEL_TAG}"
            prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
            
            think_ids = tokenizer.encode(think_part, add_special_tokens=False) if think_part else []
            answer_ids = tokenizer.encode(answer_part, add_special_tokens=False)
            
            # To compute Z, we need the text of the Answer.
            # We will embed the first chunk of the Answer (or the whole answer if short).
            # Let's align with Phase 7: Z represents ~128 tokens.
            # If answer is long, maybe multiple Zs?
            # For GSM8K, answers are usually short enough or one Z is fine.
            # Let's use the first 128 tokens of the answer as the target Z.
            
            target_text_for_z = answer_part[:500] # Approximate char limit for 128 tok
            texts_to_embed.append(target_text_for_z)
            map_text_to_sample.append(b_i)
            
            current_batch_data.append({
                'prompt_ids': prompt_ids,
                'think_ids': think_ids,
                'answer_ids': answer_ids,
                'z_vector': None
            })

        # Compute Z
        if texts_to_embed:
            with torch.no_grad():
                emb_bs = 64
                for j in range(0, len(texts_to_embed), emb_bs):
                    sub_texts = texts_to_embed[j : j+emb_bs]
                    inputs = tokenizer(sub_texts, return_tensors="pt", padding=True, truncation=True, max_length=CHUNK_SIZE).to(device)
                    
                    seq_lens = inputs.attention_mask.sum(dim=1) - 1
                    out = model.get_base_model()(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, output_hidden_states=True)
                    last_h = out.hidden_states[-1]
                    batch_h = last_h[torch.arange(len(sub_texts), device=device), seq_lens, :]
                    
                    batch_z = hypernet(batch_h.float())
                    batch_z = F.normalize(batch_z, p=2, dim=1)
                    
                    for k, z in enumerate(batch_z):
                        global_idx = j + k
                        b_i = map_text_to_sample[global_idx]
                        current_batch_data[b_i]['z_vector'] = z.cpu()
        
        # Assemble Segments
        for item in current_batch_data:
            if item['z_vector'] is None: continue
            
            # Format: [Prompt] -> [Think] -> [Z_Target] -> [Answer]
            # Note: We do NOT insert <ref> tags in the dataset for Z prediction in previous phases?
            # In Phase 8: We used 'z_target' segment type. Collator handles it.
            # We assume model predicts Z at the end of input.
            
            segments = []
            
            # 1. Prompt
            segments.append({'type': 'text', 'ids': item['prompt_ids']})
            
            # 2. Think
            if item['think_ids']:
                segments.append({'type': 'text', 'ids': item['think_ids']})
                
            # 3. Z Target (at end of Think)
            segments.append({'type': 'z_target', 'vector': item['z_vector']})
            
            # 4. Answer (Teacher Forcing)
            segments.append({'type': 'text', 'ids': item['answer_ids']})
            
            processed_samples.append(segments)

    print(f"Saving {len(processed_samples)} samples...")
    torch.save(processed_samples, OUTPUT_FILE)
    print("Done.")

if __name__ == "__main__":
    main()
