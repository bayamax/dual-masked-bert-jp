"""
Phase 7 Revised: CoT Continuation Training

Trains TinyLlama to continue reasoning by:
1. Receiving z_prev (compressed history)
2. Receiving a reference chunk (raw tokens from earlier in the CoT)
3. Receiving the current chunk
4. Predicting the next chunk

Search scope: Only chunks BEFORE the current generation point
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
import argparse
import random

# Configuration
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
CHUNK_SIZE = 127
HYPERNET_DIM = 2048
BATCH_SIZE = 2
LEARNING_RATE = 2e-4
TEMPERATURE = 0.07

class HyperNetHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.proj(x)

class CotChunkDataset(Dataset):
    def __init__(self, data_file):
        self.samples = torch.load(data_file)
        
        # Filter samples with at least 3 chunks (need prev, current, next)
        self.samples = [s for s in self.samples if s['num_chunks'] >= 3]
        
        print(f"Loaded {len(self.samples)} samples with 3+ chunks")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        chunks = sample['chunks']
        
        # Pick a random position (need at least one before and one after)
        # current_idx: the chunk we're generating
        # ref_idx: a chunk from BEFORE current_idx to use as reference
        # target: the next chunk after current
        
        if len(chunks) < 3:
            # Fallback
            current_idx = 1
        else:
            current_idx = random.randint(1, len(chunks) - 2)
        
        # Reference: any chunk before current (sliding window constraint)
        ref_idx = random.randint(0, current_idx - 1)
        
        return {
            'ref_chunk': chunks[ref_idx],
            'current_chunk': chunks[current_idx],
            'next_chunk': chunks[current_idx + 1] if current_idx + 1 < len(chunks) else chunks[current_idx],
            'z_prev': chunks[current_idx]['z_vector'],  # z up to current
        }

def collate_fn(batch):
    return batch

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load tokenizer and model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager"
    ).to(device)
    
    # LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05
    )
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()
    
    # HyperNet
    hypernet = HyperNetHead(model.config.hidden_size, HYPERNET_DIM).to(device).float()
    
    # Dataset
    dataset = CotChunkDataset(args.data_file)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    
    optimizer = torch.optim.AdamW([
        {'params': model.parameters()},
        {'params': hypernet.parameters()}
    ], lr=LEARNING_RATE)
    
    model.train()
    hypernet.train()
    
    print("Starting training...")
    
    for epoch in range(args.epochs):
        total_loss = 0
        total_gen_loss = 0
        total_ret_loss = 0
        
        for step, batch in enumerate(dataloader):
            bs = len(batch)
            
            # Build input:
            # [z_prev embedding] [reference chunk tokens] [current chunk tokens]
            # Target: [next chunk tokens]
            
            all_input_ids = []
            all_labels = []
            
            for item in batch:
                ref_tokens = item['ref_chunk']['token_ids']
                current_tokens = item['current_chunk']['token_ids']
                next_tokens = item['next_chunk']['token_ids']
                
                # Pad to CHUNK_SIZE
                ref_padded = ref_tokens + [tokenizer.pad_token_id] * (CHUNK_SIZE - len(ref_tokens))
                current_padded = current_tokens + [tokenizer.pad_token_id] * (CHUNK_SIZE - len(current_tokens))
                next_padded = next_tokens + [tokenizer.pad_token_id] * (CHUNK_SIZE - len(next_tokens))
                
                # Input: ref + current
                input_ids = ref_padded + current_padded
                
                # Labels: -100 for ref+current, next_tokens for the target
                labels = [-100] * len(input_ids) + next_padded
                input_ids = input_ids + next_padded
                
                all_input_ids.append(input_ids)
                all_labels.append(labels)
            
            # Tensorize
            input_ids = torch.tensor(all_input_ids, device=device)
            labels = torch.tensor(all_labels, device=device)
            
            # Forward: Generation loss
            outputs = model(input_ids=input_ids, labels=labels)
            loss_gen = outputs.loss
            
            # Retrieval loss: z of current should match z of ref (simplified)
            # This encourages semantic similarity encoding
            z_refs = torch.stack([item['ref_chunk']['z_vector'] for item in batch]).to(device)
            z_currents = torch.stack([item['current_chunk']['z_vector'] for item in batch]).to(device)
            
            z_refs_norm = F.normalize(z_refs, p=2, dim=1)
            z_currents_norm = F.normalize(z_currents, p=2, dim=1)
            
            # Contrastive: current should be able to find its ref
            logits = torch.mm(z_currents_norm, z_refs_norm.t()) / TEMPERATURE
            labels_ret = torch.arange(bs, device=device)
            loss_ret = F.cross_entropy(logits, labels_ret)
            
            loss = loss_gen + 0.1 * loss_ret
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_gen_loss += loss_gen.item()
            total_ret_loss += loss_ret.item()
            
            if step % 10 == 0:
                print(f"Epoch {epoch} Step {step} | Total: {loss.item():.4f} | Gen: {loss_gen.item():.4f} | Ret: {loss_ret.item():.4f}")
            
            if args.max_steps > 0 and step >= args.max_steps:
                break
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} | Avg Loss: {avg_loss:.4f}")
        
        # Save
        model.save_pretrained(f"phase7_revised_lora_epoch{epoch}")
        torch.save(hypernet.state_dict(), f"phase7_revised_hypernet_epoch{epoch}.pt")
        print(f"Saved checkpoints for epoch {epoch}")
    
    print("Training complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default="phase7_cot_chunks.pt")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=-1)
    args = parser.parse_args()
    train(args)
