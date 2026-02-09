"""
Phase 7.6: Online Retrieval Correction SFT (Dynamic Hard Negative Mining)

Trains the model to correct its retrieval errors "On-the-Fly".
1. For each batch of questions, the model performs a retrieval search against the full index.
2. If it retrieves the WRONG chunk (Top-1 Mistake), that specific wrong chunk is used as input context.
3. The model is then trained to output the CORRECT Z-vector given this "Correction Context".

Input: Question + [Top-1 Mistake]
Target: [Correct Z-Vector]
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
import time

# Configuration
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
CHUNK_SIZE = 127
HYPERNET_DIM = 2048
BATCH_SIZE = 16
ACCUM_STEPS = 4
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

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Load Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading base model & LoRA from {args.resume_lora}")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager"
    ).to(device)
    
    model = PeftModel.from_pretrained(base_model, args.resume_lora, is_trainable=True)
    model.tokenizer = tokenizer
    
    hypernet = HyperNetHead(model.config.hidden_size, HYPERNET_DIM).to(device).float()
    print(f"Loading HyperNet from {args.resume_hypernet}")
    hypernet.load_state_dict(torch.load(args.resume_hypernet, map_location=device))
    
    model.gradient_checkpointing_enable()
    
    optimizer = torch.optim.AdamW([
        {'params': model.parameters()},
        {'params': hypernet.parameters()}
    ], lr=LEARNING_RATE)
    
    # 2. Load Data & Build Index
    print(f"Loading Data: {args.data_file}")
    all_samples = torch.load(args.data_file)
    # Use only samples with multiple chunks
    all_samples = [s for s in all_samples if s['num_chunks'] >= 2]
    
    # Build In-Memory Index for Fast Retrieval
    print("Building Index...")
    all_chunks_meta = [] # Flat list of all chunks
    all_z_list = []
    
    # Map sample_idx to its correct start chunk index in all_chunks_meta
    sample_to_start_chunk_idx = {}
    
    for s_idx, sample in enumerate(all_samples):
        # We process Q -> C0 (Start Chunk) primarily
        # So we need to enable searching against ALL chunks 
        
        for c in sample['chunks']:
            global_idx = len(all_chunks_meta)
            all_chunks_meta.append({
                'token_ids': c['token_ids'],
                'chunk_index': c['chunk_index'],
                'sample_idx': s_idx,
                'question_text': sample['question'] # Store for fallback
            })
            all_z_list.append(c['z_vector']) # CPU tensor
            
            if c['chunk_index'] == 0:
                sample_to_start_chunk_idx[s_idx] = global_idx

    all_z_matrix = torch.stack(all_z_list).to(device).float()
    all_z_matrix = F.normalize(all_z_matrix, p=2, dim=1)
    
    print(f"Indexed {len(all_chunks_meta)} chunks.")
    
    # 3. Training Loop (Manual Batching)
    print("Starting Online Correction Training...")
    
    model.train()
    hypernet.train()
    optimizer.zero_grad()
    
    # Indices of samples to train on
    sample_indices = list(range(len(all_samples)))
    
    for epoch in range(args.epochs):
        random.shuffle(sample_indices)
        total_loss = 0
        correct_retrievals = 0 # How often did it get it right *initially*?
        corrected_retrievals = 0 # How often did it get right *after* correction?
        
        # Determine Batch Size
        num_batches = len(sample_indices) // BATCH_SIZE
        
        for step in range(num_batches):
            start_idx = step * BATCH_SIZE
            batch_idxs = sample_indices[start_idx : start_idx + BATCH_SIZE]
            
            # --- PHASE 1: RETRIEVAL (No Grad) ---
            # Run model on Questions to find what it retrieves
            
            questions = [all_samples[i]['question'] for i in batch_idxs]
            inputs = tokenizer(questions, return_tensors="pt", padding=True, truncation=True, max_length=CHUNK_SIZE)
            input_ids = inputs.input_ids.to(device)
            
            with torch.no_grad():
                outputs = model.get_base_model()(input_ids=input_ids, output_hidden_states=True)
                last = outputs.hidden_states[-1][:, -1, :] 
                z_q = hypernet(last.float())
                z_q_norm = F.normalize(z_q, p=2, dim=1)
                
                # Search against database
                # [B, TotalChunks]
                sim_scores = torch.mm(z_q_norm, all_z_matrix.t())
                top_scores, top_indices = torch.topk(sim_scores, k=1, dim=1)
                top_indices = top_indices.cpu().numpy().flatten()
            
            # --- PHASE 2: CONSTRUCT CORRECTION BATCH ---
            # Identify mistakes and build training batch
            
            train_input_ids = []
            train_labels = []
            train_target_z = []
            train_last_token_indices = []
            
            for b_i, s_global_idx in enumerate(batch_idxs):
                s_idx = s_global_idx # Local index into all_samples
                retrieved_idx = top_indices[b_i]
                
                correct_idx = sample_to_start_chunk_idx[s_idx]
                
                # Check if correct (Top-1 == Correct)
                if retrieved_idx == correct_idx:
                    # It got it right! Good job.
                    # We can skip correction training for this sample, OR
                    # randomly inject a "same-doc distractor" to reinforce robustness (optional)
                    correct_retrievals += 1
                    
                    # Optional: 10% chance to force train on a random distractor anyway
                    if random.random() > 0.1:
                         continue
                         
                    # Pick random distractor from same doc
                    sample_chunks = all_samples[s_idx]['chunks']
                    distractor_c = random.choice([c for c in sample_chunks if c['chunk_index'] != 0])
                    distractor_tokens = distractor_c['token_ids']
                    
                else:
                    # MISTAKE! Use this retrieved chunk as the distractor context
                    distractor_tokens = all_chunks_meta[retrieved_idx]['token_ids']
                
                # Build Training Input
                # Context: Q + <think><ref>WRONG_CHUNK</ref>
                # Target: Correct Z (all_z_matrix[correct_idx])
                
                q_text = all_samples[s_idx]['question']
                prefix_text = f"{USER_TAG}{q_text}{USER_END}{MODEL_TAG}\n{THINK_TAG}{REF_TAG}"
                
                prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
                suffix_ids = tokenizer.encode(REF_END, add_special_tokens=False)
                context_ids = prefix_ids + distractor_tokens + suffix_ids
                
                # Append Correct Generation Target (Start Chunk)
                target_chunk = all_samples[s_idx]['chunks'][0]
                target_ids = target_chunk['token_ids'] + [tokenizer.eos_token_id]
                
                full_input = context_ids + target_ids
                labels = [-100] * len(context_ids) + target_ids
                
                train_input_ids.append(full_input)
                train_labels.append(labels)
                train_target_z.append(all_z_matrix[correct_idx]) # Reuse GPU tensor
                train_last_token_indices.append(len(context_ids) - 1)
            
            if not train_input_ids:
                continue # Batch was perfect?
                
            # --- PHASE 3: TRAINING STEP ---
            
            # Pad Batch
            max_len = max(len(ids) for ids in train_input_ids)
            # Cap at 2048
            if max_len > 2048: max_len = 2048
            
            padded_inputs = []
            padded_labels = []
            
            for ids, lbls in zip(train_input_ids, train_labels):
                pad_len = max_len - len(ids)
                if pad_len >= 0:
                    padded_inputs.append(ids + [tokenizer.pad_token_id] * pad_len)
                    padded_labels.append(lbls + [-100] * pad_len)
                else:
                    padded_inputs.append(ids[-max_len:])
                    padded_labels.append(lbls[-max_len:])
            
            input_tensor = torch.tensor(padded_inputs, device=device)
            label_tensor = torch.tensor(padded_labels, device=device)
            
            # Forward
            outputs = model(input_ids=input_tensor, labels=label_tensor, output_hidden_states=True)
            loss_gen = outputs.loss
            
            # Retrieval Loss
            last_hidden_states = outputs.hidden_states[-1]
            z_preds = []
            
            # Extract Z at REF_END
            for i, idx in enumerate(train_last_token_indices):
                if idx >= last_hidden_states.size(1): idx = last_hidden_states.size(1) - 1
                h = last_hidden_states[i, idx, :]
                z_preds.append(hypernet(h.float()))
            
            z_q_batch = torch.stack(z_preds)
            z_q_norm = F.normalize(z_q_batch, p=2, dim=1)
            z_target_batch = torch.stack(train_target_z).to(device) # Should be GPU already?
            
            # Batch Contrastive Loss
            # Positive similarity
            pos_sim = torch.sum(z_q_norm * z_target_batch, dim=1)
            logits = pos_sim / TEMPERATURE
            
            # Simple loss: Maximize positive similarity (Since we don't construct explicit batch negatives here efficiently)
            # Or use InfoNCE if we treat other batch items as negatives
            # Let's use InfoNCE against other items in batch
            
            all_target_batch = torch.stack(train_target_z)
            batch_logits = torch.mm(z_q_norm, all_target_batch.t()) / TEMPERATURE
            batch_labels = torch.arange(len(z_q_batch), device=device)
            loss_ret = F.cross_entropy(batch_logits, batch_labels)
            
            loss = loss_gen + 0.5 * loss_ret
            loss = loss / ACCUM_STEPS
            
            loss.backward()
            
            if (step + 1) % ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * ACCUM_STEPS
            
            if step % (10 * ACCUM_STEPS) == 0:
                print(f"Epoch {epoch} Step {step} | Loss: {loss.item() * ACCUM_STEPS:.4f} | Initial Acc: {correct_retrievals / ((step+1)*BATCH_SIZE):.2%}")
            
            # Explicit Cleanup
            del outputs, loss, loss_gen, loss_ret, z_q_batch, z_target_batch, full_input
            del input_tensor, label_tensor
            torch.cuda.empty_cache()
        
        print(f"Epoch {epoch} Done. Model Saved.")
        model.save_pretrained(f"{args.output_dir}_epoch{epoch}")
        torch.save(hypernet.state_dict(), f"{args.output_dir}_epoch{epoch}/hypernet.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Default to Phase 7 CoT Chunks for Online Mining
    parser.add_argument("--data_file", type=str, default="phase7_cot_chunks.pt")
    parser.add_argument("--resume_lora", type=str, default="phase7_accuracy_boost_best/lora")
    parser.add_argument("--resume_hypernet", type=str, default="phase7_accuracy_boost_best/hypernet.pt")
    parser.add_argument("--output_dir", type=str, default="phase7_online_correction")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)
    train(args)
