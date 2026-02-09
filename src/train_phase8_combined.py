"""
Phase 8: Combined Reasoning + Retrieval + Correction Training (SFT)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os
import argparse

# Config
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
INIT_LORA_PATH = "phase7_online_correction_epoch2"
INIT_HYPERNET_PATH = "phase7_online_correction_epoch2/hypernet.pt"

OUTPUT_DIR = "phase8_combined_model_debug"
MAX_LENGTH = 2048

# Hyperparameters
BATCH_SIZE = 2 # Reduced from 4
ACCUM_STEPS = 16 # Increased from 8 to maintain effective batch size 32
LEARNING_RATE = 1e-4
EPOCHS = 1
ALPHA_RET = 0.5

class HyperNetHead(nn.Module):
    def __init__(self, input_dim=2048, output_dim=2048):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.proj(x)

class AugmentedDataset(Dataset):
    def __init__(self, data_path):
        print(f"Loading {data_path}...")
        self.data = torch.load(data_path)
        print(f"Loaded {len(self.data)} samples.")
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    # batch is list of segment_lists
    batch_input_ids = []
    batch_z_targets = []
    
    max_len_in_batch = 0
    pad_id = 2 # EOS token ID usually
    
    for b_i, segments in enumerate(batch):
        input_ids = []
        # z_targets: list of (index, vector)
        
        for seg in segments:
            if seg['type'] == 'text':
                ids = seg['ids']
                input_ids.extend(ids)
            elif seg['type'] == 'z_target':
                idx = len(input_ids) - 1
                if idx >= 0:
                    vec = seg['vector']
                    batch_z_targets.append((b_i, idx, vec))
        
        # Truncate
        if len(input_ids) > MAX_LENGTH:
            input_ids = input_ids[:MAX_LENGTH]
            batch_z_targets = [t for t in batch_z_targets if not (t[0]==b_i and t[1]>=MAX_LENGTH)]
            
        batch_input_ids.append(torch.tensor(input_ids, dtype=torch.long))
        max_len_in_batch = max(max_len_in_batch, len(input_ids))
    
    padded_inputs = torch.nn.utils.rnn.pad_sequence(batch_input_ids, batch_first=True, padding_value=pad_id)
    
    B, L = padded_inputs.shape
    z_gt = torch.zeros(B, L, 2048)
    z_mask = torch.zeros(B, L)
    
    for b_i, idx, vec in batch_z_targets:
        if idx < L and b_i < B:
            z_gt[b_i, idx] = vec
            z_mask[b_i, idx] = 1.0
            
    labels = padded_inputs.clone()
    labels[padded_inputs == pad_id] = -100
    
    return padded_inputs, labels, z_gt, z_mask

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Model
    print("Loading Model...")
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, device_map=device)
    model = PeftModel.from_pretrained(base, INIT_LORA_PATH, is_trainable=True)
    model.print_trainable_parameters()
    
    # Enable Gradient Checkpointing
    base.gradient_checkpointing_enable()
    model.enable_input_require_grads() 
    
    hypernet = HyperNetHead().to(device)
    hypernet.load_state_dict(torch.load(INIT_HYPERNET_PATH, map_location=device))
    hypernet.train()
    
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(hypernet.parameters()), lr=LEARNING_RATE)
    
    dataset = AugmentedDataset(args.data_file)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    
    print(f"Start Training...")
    global_step = 0
    
    torch.cuda.empty_cache()
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        
        for step, batch in enumerate(dataloader):
            input_ids, labels, z_gt, z_mask = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            z_gt = z_gt.to(device)
            z_mask = z_mask.to(device)
            
            outputs = model(input_ids=input_ids, labels=labels, output_hidden_states=True)
            loss_gen = outputs.loss
            
            last_hidden = outputs.hidden_states[-1]
            
            # Vectorized Z loss
            active_mask = z_mask > 0.5
            if active_mask.any():
                target_vecs = z_gt[active_mask]
                pred_vecs_h = last_hidden[active_mask]
                pred_z = F.normalize(hypernet(pred_vecs_h.float()), p=2, dim=1)
                loss_ret = F.mse_loss(pred_z, target_vecs.float())
            else:
                loss_ret = torch.tensor(0.0, device=device)
            
            loss = loss_gen + ALPHA_RET * loss_ret
            loss = loss / ACCUM_STEPS
            loss.backward()
            
            if (step + 1) % ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                
                if global_step % 10 == 0:
                    print(f"Step {global_step} | Loss: {loss.item()*ACCUM_STEPS:.4f} (Gen: {loss_gen.item():.4f}, Ret: {loss_ret.item():.4f})")
        
        print(f"Epoch {epoch} Done. Saving...")
        model.save_pretrained(f"{OUTPUT_DIR}_epoch{epoch}")
        torch.save(hypernet.state_dict(), f"{OUTPUT_DIR}_epoch{epoch}/hypernet.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default="phase8_augmented_dataset.pt")
    args = parser.parse_args()
    train(args)
