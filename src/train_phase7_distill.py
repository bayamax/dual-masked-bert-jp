import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import get_peft_model, LoraConfig, TaskType

# Configuration
STUDENT_MODEL_PATH = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
HYPERNET_DIM = 2048
CHUNK_SIZE = 127
BATCH_SIZE = 4
LEARNING_RATE = 2e-5
EPOCHS = 3
TEMPERATURE = 0.07 # From Phase 6

class AttentionDistillationDataset(Dataset):
    def __init__(self, data_file):
        self.samples = []
        with open(data_file, 'r') as f:
            for line in f:
                try:
                    self.samples.append(json.loads(line))
                except: continue
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

def custom_collate_fn(batch):
    # batch is a list of dicts
    query_texts = [item['query_text'] for item in batch]
    top_ref_texts = [item['top_ref_text'] for item in batch]
    
    # Handle variable length chunk probs
    # We need to pad them to the max length in this batch
    chunk_probs_list = [item['label_chunk_probs'] for item in batch]
    max_len = max(len(p) for p in chunk_probs_list)
    
    # Pad with 0.0 (or -1.0 for ignore)
    # Since these are probabilities summing to 1, 0.0 is safe for "no mass".
    # We will create a mask if needed, but for now just padding.
    padded_probs = torch.zeros(len(batch), max_len)
    for i, p in enumerate(chunk_probs_list):
        l = len(p)
        padded_probs[i, :l] = torch.tensor(p)
        
    return {
        "query_text": query_texts,
        "top_ref_text": top_ref_texts,
        "label_chunk_probs": padded_probs
    }

class HyperNetHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # Simple projection as per Phase 6 / HyperPromptNet structure
        self.proj = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.proj(x)

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Load Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_PATH)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL_PATH, 
        torch_dtype=torch.bfloat16,
        attn_implementation="eager" # Required for output_attentions=True
    ).to(device)
    
    # LoRA Setup for Student
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"])
    model = get_peft_model(base_model, peft_config)
    
    # ... (snip) ...

            # 4. Attention Balance Loss
            full_texts = [f"{r}\n{q}" for r, q in zip(top_ref_texts, query_texts)]
            inputs_full = tokenizer(full_texts, return_tensors="pt", padding=True).to(device)
            # Pass labels to calculate loss
            outputs_full = model(
                inputs_full.input_ids, 
                attention_mask=inputs_full.attention_mask, 
                labels=inputs_full.input_ids,
                output_attentions=True
            )
            loss_gen = outputs_full.loss
            
            loss = loss_retrieval + loss_gen
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if step % 1 == 0:
                print(f"Epoch {epoch} Step {step} Loss: {loss.item():.4f} (Ret: {loss_retrieval.item():.4f})")
            
            if args.max_steps > 0 and step >= args.max_steps:
                print("Max steps reached (Debug mode). Stopping epoch.")
                break

        # Save
        torch.save(hypernet_head.state_dict(), f"phase7_hypernet_epoch{epoch}.pt")
        model.save_pretrained(f"phase7_lora_epoch{epoch}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default="phase7_attention_distill.jsonl")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=-1, help="Stop after N steps per epoch for debugging")
    args = parser.parse_args()
    EPOCHS = args.epochs
    train(args)
