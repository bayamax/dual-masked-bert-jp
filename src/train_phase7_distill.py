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
    
    # Fix: Use eager attention for compatibility with output_attentions=True
    base_model = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL_PATH, 
        torch_dtype=torch.bfloat16,
        attn_implementation="eager"
    ).to(device)
    
    # LoRA Setup for Student
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"])
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()
    
    # HyperNet Setup
    hypernet_head = HyperNetHead(model.config.hidden_size, HYPERNET_DIM).to(device).float()
    
    optimizer = torch.optim.AdamW([
        {'params': model.parameters()},
        {'params': hypernet_head.parameters()}
    ], lr=LEARNING_RATE)
    
    dataset = AttentionDistillationDataset(args.data_file)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
    
    criterion_kl = nn.KLDivLoss(reduction='batchmean')
    
    model.train()
    hypernet_head.train()
    
    print("Starting Training...")
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for step, batch in enumerate(dataloader):
            # Batch items
            query_texts = batch['query_text'] 
            top_ref_texts = batch['top_ref_text']
            # chunk_probs is available but skipping explicit usage logic for this prototype step
            # chunk_probs = batch['label_chunk_probs'].to(device)
            
            # 1. Embed Query
            inputs_q = tokenizer(query_texts, return_tensors="pt", padding=True, truncation=True).to(device)
            outputs_q = model.get_base_model()(inputs_q.input_ids, attention_mask=inputs_q.attention_mask, output_hidden_states=True)
            last_hidden_q = outputs_q.hidden_states[-1][:, -1, :] # [B, H]
            z_query = hypernet_head(last_hidden_q.float()) # [B, Dim]
            
            # 2. Embed Ref (The Target/Key) using Recursive Logic
            inputs_ref = tokenizer(top_ref_texts, return_tensors="pt", padding=True, truncation=True, max_length=CHUNK_SIZE).to(device)
            bs = len(top_ref_texts)
            z_prev_dummy = torch.zeros(bs, 1, model.config.hidden_size, device=device).bfloat16()
            
            ref_embeds = model.get_base_model().model.embed_tokens(inputs_ref.input_ids)
            combined_embeds = torch.cat([z_prev_dummy, ref_embeds], dim=1)
            
            out_ref = model.get_base_model()(inputs_embeds=combined_embeds, output_hidden_states=True)
            last_hidden_ref = out_ref.hidden_states[-1][:, -1, :] 
            z_ref = hypernet_head(last_hidden_ref.float()) 
            
            # 3. Retrieval Loss (Contrastive/InfoNCE)
            z_q_norm = F.normalize(z_query, p=2, dim=1)
            z_ref_norm = F.normalize(z_ref, p=2, dim=1)
            
            logits = torch.mm(z_q_norm, z_ref_norm.t()) / TEMPERATURE
            labels = torch.arange(bs, device=device)
            loss_retrieval = F.cross_entropy(logits, labels)
            
            # 4. Attention Balance Loss
            full_texts = [f"{r}\n{q}" for r, q in zip(top_ref_texts, query_texts)]
            inputs_full = tokenizer(full_texts, return_tensors="pt", padding=True).to(device)
            
            # Fix: Pass labels to calculate loss, include output_attentions=True
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
                print(f"Epoch {epoch} Step {step} Loss: {loss.item():.4f} (Ret: {loss_retrieval.item():.4f}, Gen: {loss_gen.item():.4f})")
            
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
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    args = parser.parse_args()
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    train(args)
