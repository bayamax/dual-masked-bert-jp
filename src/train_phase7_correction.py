"""
Phase 7.6: Retrieval Correction SFT (Iterative Retrieval Training)

Trains the model to correct its retrieval when given a "Wrong" chunk.
The goal is to teach the model: "If the retrieved context is helpful but not the *start* (or correct next step), use it to find the *actual* correct step."

This addresses the "Context Misalignment" issue where the model retrieves a later chunk (e.g., C3) instead of the start (C0).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import argparse
import random
import os
import json

# Configuration
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
CHUNK_SIZE = 127
HYPERNET_DIM = 2048
BATCH_SIZE = 64
LEARNING_RATE = 2e-4
TEMPERATURE = 0.07

# Special Tokens
USER_TAG = "<user>"
USER_END = "</user>\n"
MODEL_TAG = "<model>"
THINK_TAG = "<think>\n"
REF_TAG = "<ref>"
REF_END = "</ref>"

class HyperNetHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.proj(x)

class CorrectionDataset(Dataset):
    def __init__(self, samples, tokenizer):
        self.samples = samples
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx] # This is now a "Hard Negative Pair" object
        
        question = sample['question']
        
        # Hard Negative Mining has already selected the best "Wrong Chunk"
        distractor_chunk = sample['hard_negative_chunk']
        distractor_tokens = distractor_chunk['token_ids']
        
        target_chunk = sample['target_chunk']
        target_tokens = target_chunk['token_ids']
        target_z = target_chunk['z_vector']
        
        # Determine correction type for logging/analysis (optional)
        # correction_type = "Context Misalignment" if distractor_chunk.get('from_same_sample', False) else "Hard Distractor"
        
        # Construction:
        # <user>Question</user><model><think><ref>WRONG_CHUNK</ref>
        # The model should output Z that matches target_z (C0)
        
        context_text = f"{USER_TAG}{question}{USER_END}{MODEL_TAG}\n{THINK_TAG}{REF_TAG}"
        
        return {
            'question': question,
            'context_prefix': context_text,
            'distractor_tokens': distractor_tokens,
            'target_tokens': target_tokens,
            'target_z': target_z
        }

def collate_fn(batch):
    return batch

def evaluate(model, hypernet, val_loader, device):
    model.eval()
    hypernet.eval()
    
    total_samples = 0
    correct_retrievals = 0
    total_loss = 0
    
    with torch.no_grad():
        for batch in val_loader:
            all_input_ids = []
            all_labels = []
            
            query_embeds_list = [] 
            target_z_list = []     
            
            for item in batch:
                # Construct Input: Prefix + Distractor + Suffix
                # We want the model to see the distractor in the context
                
                prefix_ids = model.tokenizer.encode(item['context_prefix'], add_special_tokens=False)
                suffix_ids = model.tokenizer.encode(REF_END, add_special_tokens=False)
                
                # Context = Prefix + Distractor + Suffix
                # We need to ensure the query embedding is computed from this *entire context* or just Q?
                # User's logic: "Reference the raw token of the wrong chunk... choose correct Z"
                # So the query should be (Last Token of Distractor Context).
                
                context_ids = prefix_ids + item['distractor_tokens'] + suffix_ids
                
                target_ids = item['target_tokens'] + [model.tokenizer.eos_token_id]
                
                input_ids = context_ids + target_ids
                labels = [-100] * len(context_ids) + target_ids
                
                pad_len = CHUNK_SIZE * 3 - len(input_ids) # Larger buffer for Context+Distractor
                if pad_len > 0:
                    input_ids = input_ids + [model.tokenizer.pad_token_id] * pad_len
                    labels = labels + [-100] * pad_len
                else:
                    input_ids = input_ids[:CHUNK_SIZE*3]
                    labels = labels[:CHUNK_SIZE*3]
                
                all_input_ids.append(input_ids)
                all_labels.append(labels)
                
                # For Retrieval Loss, we use the embedding of the last token of 'context_ids'
                # But our HyperNet takes hidden states. So we need to compute full forward.
                # We mark this for the loop below.
                query_embeds_list.append(torch.tensor(context_ids, device=device)) 
                target_z_list.append(item['target_z'])
                
            input_ids = torch.stack([torch.tensor(ids, device=device) for ids in all_input_ids]) # Stack properly
            labels = torch.stack([torch.tensor(lbls, device=device) for lbls in all_labels])
            
            # Forward for Generation Loss
            if input_ids.size(1) > 2048: # Safety truncation
                 input_ids = input_ids[:, -2048:]
                 labels = labels[:, -2048:]

            outputs = model(input_ids=input_ids, labels=labels, output_hidden_states=True)
            loss_gen = outputs.loss
            
            # Forward for Retrieval Loss
            loss_ret = torch.tensor(0.0, device=device)
            if len(query_embeds_list) > 0:
                 # Extract the hidden state at the END of the context (before generation starts)
                 # We need to find the index of the last token of context.
                 # Since we padded, we can use the original lengths.
                 
                 # Optimization: Utilizing the 'outputs' from generation?
                 # outputs.hidden_states is a tuple of (L, B, S, H). We want last layer.
                 last_hidden_states = outputs.hidden_states[-1] # [B, S, H]
                 
                 # We need Z from the position corresponding to REF_END
                 # In 'input_ids', this is right before 'target_ids' start.
                 # Let's find the effective sequence length of context for each item.
                 
                 z_preds = []
                 for i in range(len(batch)):
                      # Calculate length of context (prefix + distractor + suffix)
                      ctx_len = len(model.tokenizer.encode(batch[i]['context_prefix'], add_special_tokens=False)) + \
                                len(batch[i]['distractor_tokens']) + \
                                len(model.tokenizer.encode(REF_END, add_special_tokens=False))
                      
                      # The Z should come from hidden state at (ctx_len - 1)
                      # Check bounds
                      if ctx_len >= last_hidden_states.size(1): ctx_len = last_hidden_states.size(1) - 1
                      h = last_hidden_states[i, ctx_len-1, :]
                      z_preds.append(hypernet(h.float()))
                 
                 z_q = torch.stack(z_preds)
                 z_q_norm = F.normalize(z_q, p=2, dim=1)
                 
                 z_targets = torch.stack([z.to(device) for z in target_z_list])
                 z_targets_norm = F.normalize(z_targets.float(), p=2, dim=1)
                 
                 logits = torch.mm(z_q_norm, z_targets_norm.t()) # [B, B]
                 labels_ret = torch.arange(len(batch), device=device)
                 
                 preds = torch.argmax(logits, dim=1)
                 correct_retrievals += (preds == labels_ret).sum().item()
                 total_samples += len(batch)
                 
                 loss_ret = F.cross_entropy(logits / TEMPERATURE, labels_ret)
            
            loss = loss_gen + 0.5 * loss_ret
            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    acc = correct_retrievals / total_samples if total_samples > 0 else 0
    return avg_loss, acc

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    # Load Model (Resume from Phase 7.5 Best)
    print(f"Loading base model & LoRA from {args.resume_lora}")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager"
    ).to(device)
    
    model = PeftModel.from_pretrained(base_model, args.resume_lora, is_trainable=True)
    model.tokenizer = tokenizer
    
    # HyperNet
    hypernet = HyperNetHead(model.config.hidden_size, HYPERNET_DIM).to(device).float()
    print(f"Loading HyperNet from {args.resume_hypernet}")
    hypernet.load_state_dict(torch.load(args.resume_hypernet, map_location=device))
    
    # Dataset
    print(f"Loading Data: {args.data_file}")
    all_samples = torch.load(args.data_file)
    all_samples = [s for s in all_samples if s['num_chunks'] >= 2]
    
    # Split
    total_len = len(all_samples)
    val_len = int(total_len * 0.1)
    train_len = total_len - val_len
    
    random.seed(42)
    random.shuffle(all_samples)
    
    train_samples = all_samples[:train_len]
    val_samples = all_samples[train_len:]
    
    train_dataset = CorrectionDataset(train_samples, tokenizer)
    val_dataset = CorrectionDataset(val_samples, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    model.gradient_checkpointing_enable()
    
    optimizer = torch.optim.AdamW([
        {'params': model.parameters()},
        {'params': hypernet.parameters()}
    ], lr=LEARNING_RATE)
    
    print("Starting Correction Training...")
    best_val_acc = 0.0
    
    for epoch in range(args.epochs):
        model.train()
        hypernet.train()
        train_loss = 0
        
        for step, batch in enumerate(train_loader):
            # ... (Copied training logic matches evaluate but with backward) ...
            
            # Construction (Same as Evaluate)
            all_input_ids = []
            all_labels = []
            target_z_list = []
            last_hidden_indices = []
            
            for item in batch:
                prefix_ids = tokenizer.encode(item['context_prefix'], add_special_tokens=False)
                suffix_ids = tokenizer.encode(REF_END, add_special_tokens=False)
                context_ids = prefix_ids + item['distractor_tokens'] + suffix_ids
                target_ids = item['target_tokens'] + [tokenizer.eos_token_id]
                
                input_ids = context_ids + target_ids
                labels = [-100] * len(context_ids) + target_ids
                
                pad_len = CHUNK_SIZE * 3 - len(input_ids)
                if pad_len > 0:
                    input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
                    labels = labels + [-100] * pad_len
                else:
                    input_ids = input_ids[:CHUNK_SIZE*3]
                    labels = labels[:CHUNK_SIZE*3]
                
                all_input_ids.append(input_ids)
                all_labels.append(labels)
                target_z_list.append(item['target_z'])
                
                # Index for Z extraction (end of context)
                ctx_len = len(context_ids)
                last_hidden_indices.append(ctx_len - 1)
            
            input_ids = torch.stack([torch.tensor(ids, device=device) for ids in all_input_ids])
            labels = torch.stack([torch.tensor(lbls, device=device) for lbls in all_labels])
            
            if input_ids.size(1) > 2048:
                 input_ids = input_ids[:, -2048:]
                 labels = labels[:, -2048:]

            outputs = model(input_ids=input_ids, labels=labels, output_hidden_states=True)
            loss_gen = outputs.loss
            
            # Retrieval Loss
            last_hidden_states = outputs.hidden_states[-1]
            z_preds = []
            for i, idx in enumerate(last_hidden_indices):
                if idx >= last_hidden_states.size(1): idx = last_hidden_states.size(1) - 1
                h = last_hidden_states[i, idx, :]
                z_preds.append(hypernet(h.float()))
            
            z_q = torch.stack(z_preds)
            z_q_norm = F.normalize(z_q, p=2, dim=1)
            z_targets = torch.stack([z.to(device) for z in target_z_list])
            z_targets_norm = F.normalize(z_targets.float(), p=2, dim=1)
            
            logits = torch.mm(z_q_norm, z_targets_norm.t()) / TEMPERATURE
            labels_ret = torch.arange(len(batch), device=device)
            loss_ret = F.cross_entropy(logits, labels_ret)
            
            loss = loss_gen + 0.5 * loss_ret
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            if step % 50 == 0:
                 print(f"Epoch {epoch} Step {step} | Loss: {loss.item():.4f}")
                 
        val_loss, val_acc = evaluate(model, hypernet, val_loader, device)
        print(f"Epoch {epoch} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f} | Correction Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"New Best Accuracy! Saving to {args.output_dir}_best...")
            if not os.path.exists(f"{args.output_dir}_best"): os.makedirs(f"{args.output_dir}_best")
            model.save_pretrained(f"{args.output_dir}_best/lora")
            torch.save(hypernet.state_dict(), f"{args.output_dir}_best/hypernet.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default="phase7_hard_negatives.pt")
    parser.add_argument("--resume_lora", type=str, default="phase7_accuracy_boost_best/lora")
    parser.add_argument("--resume_hypernet", type=str, default="phase7_accuracy_boost_best/hypernet.pt")
    parser.add_argument("--output_dir", type=str, default="phase7_correction_output")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    train(args)
