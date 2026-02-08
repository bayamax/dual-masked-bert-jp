"""
Phase 7.5: CoT Continuation Training (DeepSeek R1 Style + Retrieval Fix + Val Split)

Trains TinyLlama to:
1. Retrieve the first chunk given the Question (Q -> C0)
2. Retrieve the next chunk given previous context (C_t-1 -> C_t)
3. Generate the next chunk with DeepSeek R1 format

Features:
- Train/Validation Split (90/10)
- Validation Metric: Start-Chunk Retrieval Accuracy (Top-1)
- Save Best Model
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

class CotChunkDataset(Dataset):
    def __init__(self, samples, tokenizer):
        self.samples = samples
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        chunks = sample['chunks']
        question = sample['question']
        
        # Two modes of training:
        # 1. Start: Q -> C0 (Retrieval + Generation)
        # 2. Continuation: C_t -> C_t+1 (Retrieval + Generation)
        
        mode = "start" if random.random() < 0.5 else "continue"
        
        if mode == "start":
            ref_chunk_tokens = [] 
            ref_z = chunks[0]['z_vector'] 
            target_chunk = chunks[0]
            current_tokens = chunks[0]['token_ids']
            context_text = f"{USER_TAG}{question}{USER_END}{MODEL_TAG}\n{THINK_TAG}{REF_TAG}"
            
        else:
            if len(chunks) < 2:
                # Fallback to start if only 1 chunk
                # Recursive call might be dangerous if all are 1 chunk, but filtered outside
                return self.__getitem__(random.randint(0, len(self.samples)-1))
                
            target_idx = random.randint(1, len(chunks) - 1)
            target_chunk = chunks[target_idx]
            current_tokens = target_chunk['token_ids']
            
            ref_idx = target_idx - 1
            ref_chunk = chunks[ref_idx]
            ref_chunk_tokens = ref_chunk['token_ids']
            
            context_text = f"{REF_TAG}" 
            ref_z = None 
        
        return {
            'mode': mode,
            'question': question,
            'context_text': context_text,
            'ref_tokens': ref_chunk_tokens,
            'target_tokens': current_tokens,
            'ref_z': ref_z, 
            'target_z': target_chunk['z_vector']
        }

def collate_fn(batch):
    return batch

def evaluate(model, hypernet, val_loader, device):
    model.eval()
    hypernet.eval()
    
    total_samples = 0
    correct_retrievals = 0
    total_loss = 0
    
    # We will build a mini-index of target Zs for the batch to calculate batch-wise accuracy
    # Real validation should index ALL chunks, but computationally expensive to do every epoch.
    # We approximate by checking if correct chunk is retrieved *among the batch negatives*.
    
    with torch.no_grad():
        for batch in val_loader:
            # Reconstruct batch logic for loss
            # ... (Copied mostly from train loop but no backward)
            
            all_input_ids = []
            all_labels = []
            
            query_embeds_list = [] 
            target_z_list = []     
            
            for item in batch:
                mode = item['mode']
                
                # Gen Input
                if mode == "start":
                    context_ids = model.tokenizer.encode(item['context_text'], add_special_tokens=False)
                    input_ids = context_ids + item['target_tokens'] + [model.tokenizer.eos_token_id]
                    labels = [-100] * len(context_ids) + item['target_tokens'] + [model.tokenizer.eos_token_id]
                    
                    q_ids = model.tokenizer.encode(item['question'], add_special_tokens=False)
                    q_ids = q_ids[:CHUNK_SIZE]
                    q_ids += [model.tokenizer.pad_token_id] * (CHUNK_SIZE - len(q_ids))
                    query_embeds_list.append(torch.tensor(q_ids, device=device))
                    target_z_list.append(item['ref_z']) 
                    
                else:
                    ref_part = model.tokenizer.encode(REF_TAG, add_special_tokens=False) + item['ref_tokens'] + model.tokenizer.encode(REF_END, add_special_tokens=False)
                    target_ids = item['target_tokens'] + [model.tokenizer.eos_token_id]
                    input_ids = ref_part + target_ids
                    labels = [-100] * len(ref_part) + target_ids
                
                pad_len = CHUNK_SIZE * 2 - len(input_ids)
                if pad_len > 0:
                    input_ids = input_ids + [model.tokenizer.pad_token_id] * pad_len
                    labels = labels + [-100] * pad_len
                else:
                    input_ids = input_ids[:CHUNK_SIZE*2]
                    labels = labels[:CHUNK_SIZE*2]
                
                all_input_ids.append(input_ids)
                all_labels.append(labels)
                
            input_ids = torch.tensor(all_input_ids, device=device)
            labels = torch.tensor(all_labels, device=device)
            
            outputs = model(input_ids=input_ids, labels=labels)
            loss_gen = outputs.loss
            
            loss_ret = torch.tensor(0.0, device=device)
            if len(query_embeds_list) > 0:
                q_input_ids = torch.stack(query_embeds_list).to(device)
                
                base_out = model.get_base_model()(input_ids=q_input_ids, output_hidden_states=True)
                last_hidden = base_out.hidden_states[-1][:, -1, :] 
                z_q = hypernet(last_hidden.float())
                z_q_norm = F.normalize(z_q, p=2, dim=1)
                
                z_targets = torch.stack([z.to(device) for z in target_z_list])
                z_targets_norm = F.normalize(z_targets.float(), p=2, dim=1)
                
                logits = torch.mm(z_q_norm, z_targets_norm.t()) # [B, B]
                labels_ret = torch.arange(len(query_embeds_list), device=device)
                
                # Accuracy (Batch-wise Top-1)
                preds = torch.argmax(logits, dim=1)
                correct_retrievals += (preds == labels_ret).sum().item()
                total_samples += len(query_embeds_list)
                
                # Scaled Logits for Loss
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
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager"
    ).to(device)
    
    # Load LoRA
    if args.resume_lora and os.path.exists(args.resume_lora):
        print(f"Resuming LoRA from {args.resume_lora}")
        model = PeftModel.from_pretrained(base_model, args.resume_lora, is_trainable=True)
    else:
        print("Starting new LoRA (if not resuming)")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05
        )
        model = get_peft_model(base_model, peft_config)
    
    model.tokenizer = tokenizer # Attach for eval
    
    # HyperNet
    hypernet = HyperNetHead(model.config.hidden_size, HYPERNET_DIM).to(device).float()
    if args.resume_hypernet and os.path.exists(args.resume_hypernet):
        print(f"Resuming HyperNet from {args.resume_hypernet}")
        hypernet.load_state_dict(torch.load(args.resume_hypernet, map_location=device))
    
    # Dataset Load & Split
    all_samples = torch.load(args.data_file)
    # Filter
    all_samples = [s for s in all_samples if s['num_chunks'] >= 2]
    
    # Split
    total_len = len(all_samples)
    val_len = int(total_len * 0.1) # 10%
    train_len = total_len - val_len
    
    # Shuffle (reproducible)
    random.seed(42)
    random.shuffle(all_samples)
    
    train_samples = all_samples[:train_len]
    val_samples = all_samples[train_len:]
    
    print(f"Dataset Split: Train={len(train_samples)}, Val={len(val_samples)}")
    
    train_dataset = CotChunkDataset(train_samples, tokenizer)
    val_dataset = CotChunkDataset(val_samples, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    model.gradient_checkpointing_enable()
    
    optimizer = torch.optim.AdamW([
        {'params': model.parameters()},
        {'params': hypernet.parameters()}
    ], lr=LEARNING_RATE)
    
    # Training Loop
    best_val_acc = 0.0
    
    print("Starting training...")
    
    for epoch in range(args.epochs):
        model.train()
        hypernet.train()
        
        train_loss = 0
        
        for step, batch in enumerate(train_loader):
            # ... (Same Train Logic as before) ...
            all_input_ids = []
            all_labels = []
            query_embeds_list = [] 
            target_z_list = []     
            
            for item in batch:
                mode = item['mode']
                if mode == "start":
                    context_ids = tokenizer.encode(item['context_text'], add_special_tokens=False)
                    input_ids = context_ids + item['target_tokens'] + [tokenizer.eos_token_id]
                    labels = [-100] * len(context_ids) + item['target_tokens'] + [tokenizer.eos_token_id]
                    q_ids = tokenizer.encode(item['question'], add_special_tokens=False)
                    q_ids = q_ids[:CHUNK_SIZE]
                    q_ids += [tokenizer.pad_token_id] * (CHUNK_SIZE - len(q_ids))
                    query_embeds_list.append(torch.tensor(q_ids, device=device))
                    target_z_list.append(item['ref_z']) 
                else:
                    ref_part = tokenizer.encode(REF_TAG, add_special_tokens=False) + item['ref_tokens'] + tokenizer.encode(REF_END, add_special_tokens=False)
                    target_ids = item['target_tokens'] + [tokenizer.eos_token_id]
                    input_ids = ref_part + target_ids
                    labels = [-100] * len(ref_part) + target_ids
                
                pad_len = CHUNK_SIZE * 2 - len(input_ids)
                if pad_len > 0:
                    input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
                    labels = labels + [-100] * pad_len
                else:
                    input_ids = input_ids[:CHUNK_SIZE*2]
                    labels = labels[:CHUNK_SIZE*2]
                all_input_ids.append(input_ids)
                all_labels.append(labels)
            
            input_ids = torch.tensor(all_input_ids, device=device)
            labels = torch.tensor(all_labels, device=device)
            
            outputs = model(input_ids=input_ids, labels=labels)
            loss_gen = outputs.loss
            
            loss_ret = torch.tensor(0.0, device=device)
            if len(query_embeds_list) > 0:
                q_input_ids = torch.stack(query_embeds_list).to(device)
                base_out = model.get_base_model()(input_ids=q_input_ids, output_hidden_states=True)
                last = base_out.hidden_states[-1][:, -1, :] 
                z_q = hypernet(last.float())
                z_q_norm = F.normalize(z_q, p=2, dim=1)
                z_targets = torch.stack([z.to(device) for z in target_z_list])
                z_targets_norm = F.normalize(z_targets.float(), p=2, dim=1)
                logits = torch.mm(z_q_norm, z_targets_norm.t()) / TEMPERATURE
                labels_ret = torch.arange(len(query_embeds_list), device=device)
                loss_ret = F.cross_entropy(logits, labels_ret)
                if torch.isnan(loss_ret): loss_ret = torch.tensor(0.0, device=device)
            
            loss = loss_gen + 0.5 * loss_ret
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            if step % 50 == 0:
                 print(f"Epoch {epoch} Step {step} | Loss: {loss.item():.4f}")

        # Validation
        val_loss, val_acc = evaluate(model, hypernet, val_loader, device)
        print(f"Epoch {epoch} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f} | Val Retrieval Acc (Batch): {val_acc:.4f}")
        
        # Save Regular
        model.save_pretrained(f"phase7_revised_lora_epoch{epoch}")
        torch.save(hypernet.state_dict(), f"phase7_revised_hypernet_epoch{epoch}.pt")
        
        # Save Best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"New Best Accuracy! Saving to {args.output_dir}_best...")
            if not os.path.exists(f"{args.output_dir}_best"): os.makedirs(f"{args.output_dir}_best")
            model.save_pretrained(f"{args.output_dir}_best/lora")
            torch.save(hypernet.state_dict(), f"{args.output_dir}_best/hypernet.pt")

    print("Training complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default="phase7_cot_chunks.pt")
    parser.add_argument("--resume_lora", type=str, required=False)
    parser.add_argument("--resume_hypernet", type=str, required=False)
    parser.add_argument("--output_dir", type=str, default="phase7_output")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    train(args)
