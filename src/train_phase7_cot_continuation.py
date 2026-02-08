"""
Phase 7 Revised: CoT Continuation Training (DeepSeek R1 Style + Retrieval Fix)

Trains TinyLlama to:
1. Retrieve the first chunk given the Question (Q -> C0)
2. Retrieve the next chunk given previous context (C_t-1 -> C_t)
3. Generate the next chunk with DeepSeek R1 format

Format:
<user>{question}</user>
<model>
<think>
<ref>{retrieved_chunk}</ref>
{generated_chunk}
...
</think>
{final_answer}
</model>
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

# Configuration
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
CHUNK_SIZE = 127
HYPERNET_DIM = 2048
BATCH_SIZE = 8
LEARNING_RATE = 2e-4
TEMPERATURE = 0.07

# Special Tokens (simulated for TinyLlama if not present)
USER_TAG = "<user>"
USER_END = "</user>\n"
MODEL_TAG = "<model>"
MODEL_END = "</model>"
THINK_TAG = "<think>\n"
THINK_END = "\n</think>"
REF_TAG = "<ref>"
REF_END = "</ref>"

class HyperNetHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.proj(x)

class CotChunkDataset(Dataset):
    def __init__(self, data_file, tokenizer):
        self.samples = torch.load(data_file)
        self.tokenizer = tokenizer
        
        # Filter samples with at least 2 chunks (Start + End at least)
        self.samples = [s for s in self.samples if s['num_chunks'] >= 2]
        
        print(f"Loaded {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        chunks = sample['chunks']
        question = sample['question']
        
        # Two modes of training:
        # 1. Start: Q -> C0 (Retrieval + Generation)
        # 2. Continuation: C_t -> C_t+1 (Retrieval + Generation)
        
        # Randomly choose mode (50/50)
        mode = "start" if random.random() < 0.5 else "continue"
        
        if mode == "start":
            # Target: Chunk 0
            # Input: <user>Q</user><model><think><ref>
            
            ref_chunk_tokens = [] # No reference for start
            ref_z = chunks[0]['z_vector'] # We want Q to match z_c0
            
            target_chunk = chunks[0]
            current_tokens = chunks[0]['token_ids']
            
            # Context for generation
            # Q is the context.
            context_text = f"{USER_TAG}{question}{USER_END}{MODEL_TAG}\n{THINK_TAG}{REF_TAG}"
            
        else:
            # Continuation
            # Pick a target chunk from 1 to End
            if len(chunks) < 2:
                # Fallback to start if only 1 chunk (should be filtered out but just in case)
                return self.__getitem__(random.randint(0, len(self.samples)-1))
                
            target_idx = random.randint(1, len(chunks) - 1)
            target_chunk = chunks[target_idx]
            current_tokens = target_chunk['token_ids']
            
            # Reference: Any chunk before target (usually immediately before or relevant)
            # For strict CoT continuation, it is usually C_t-1.
            ref_idx = target_idx - 1
            ref_chunk = chunks[ref_idx]
            ref_chunk_tokens = ref_chunk['token_ids']
            
            # Simulated Context: ...<ref>REF_CHUNK</ref>
            # Note: We don't have full history, so we rely on finding REF_TAG as trigger
            context_text = f"{REF_TAG}" 
            
            # We don't train retrieval for continuation here (keeping it simple for now)
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

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load tokenizer
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager"
    ).to(device)
    
    # Load LoRA (Resume or New)
    if args.resume_lora and os.path.exists(args.resume_lora):
        print(f"Resuming LoRA from {args.resume_lora}")
        model = PeftModel.from_pretrained(base_model, args.resume_lora, is_trainable=True)
    else:
        print("Starting new LoRA (if not resuming)")
        if args.resume_lora: print(f"Warning: {args.resume_lora} not found, starting new.")
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
    if args.resume_hypernet and os.path.exists(args.resume_hypernet):
        print(f"Resuming HyperNet from {args.resume_hypernet}")
        hypernet.load_state_dict(torch.load(args.resume_hypernet, map_location=device))
    elif args.resume_hypernet:
        print(f"Warning: {args.resume_hypernet} not found, starting new.")
    
    # Dataset
    dataset = CotChunkDataset(args.data_file, tokenizer)
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
            all_input_ids = []
            all_labels = []
            
            # Retrieval lists
            query_embeds_list = [] 
            target_z_list = []     
            
            for item in batch:
                mode = item['mode']
                
                # --- Generation Input Construction ---
                if mode == "start":
                    # <user>Q</user><model><think><ref> -> [TargetChunk]
                    context_ids = tokenizer.encode(item['context_text'], add_special_tokens=False)
                    input_ids = context_ids + item['target_tokens'] + [tokenizer.eos_token_id]
                    # Label: Mask context, predict target
                    labels = [-100] * len(context_ids) + item['target_tokens'] + [tokenizer.eos_token_id]
                    
                    # Store for retrieval loss
                    q_ids = tokenizer.encode(item['question'], add_special_tokens=False)
                    q_ids = q_ids[:CHUNK_SIZE]
                    q_ids += [tokenizer.pad_token_id] * (CHUNK_SIZE - len(q_ids))
                    query_embeds_list.append(torch.tensor(q_ids, device=device))
                    target_z_list.append(item['ref_z']) # z_c0
                    
                else:
                    # [Ref_Id] + [Ref_Tokens] + [Ref_End] + [Target]
                    # context_text was just <ref>
                    # We need to construct <ref>TOKENS</ref>
                    ref_part = tokenizer.encode(REF_TAG, add_special_tokens=False) + item['ref_tokens'] + tokenizer.encode(REF_END, add_special_tokens=False)
                    target_ids = item['target_tokens'] + [tokenizer.eos_token_id]
                    
                    input_ids = ref_part + target_ids
                    labels = [-100] * len(ref_part) + target_ids
                
                # Pad
                pad_len = CHUNK_SIZE * 2 - len(input_ids)
                if pad_len > 0:
                    input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
                    labels = labels + [-100] * pad_len
                else:
                    input_ids = input_ids[:CHUNK_SIZE*2]
                    labels = labels[:CHUNK_SIZE*2]
                
                all_input_ids.append(input_ids)
                all_labels.append(labels)
            
            # Tensorize Gen Input
            input_ids = torch.tensor(all_input_ids, device=device)
            labels = torch.tensor(all_labels, device=device)
            
            # Forward Gen
            outputs = model(input_ids=input_ids, labels=labels)
            loss_gen = outputs.loss
            
            # Retrieval Loss (Only for Start Mode)
            loss_ret = torch.tensor(0.0, device=device)
            
            if len(query_embeds_list) > 0:
                q_input_ids = torch.stack(query_embeds_list).to(device)
                
                # Encode Q -> z_q
                base_out = model.get_base_model()(input_ids=q_input_ids, output_hidden_states=True)
                last_hidden = base_out.hidden_states[-1][:, -1, :] 
                z_q = hypernet(last_hidden.float())
                z_q_norm = F.normalize(z_q, p=2, dim=1)
                
                # Target Z (Precomputed C0)
                z_targets = torch.stack([z.to(device) for z in target_z_list])
                z_targets_norm = F.normalize(z_targets.float(), p=2, dim=1)
                
                # Contrastive Loss
                logits = torch.mm(z_q_norm, z_targets_norm.t()) / TEMPERATURE
                labels_ret = torch.arange(len(query_embeds_list), device=device)
                
                loss_ret = F.cross_entropy(logits, labels_ret)
            
            if torch.isnan(loss_ret): loss_ret = torch.tensor(0.0, device=device)
            
            loss = loss_gen + 0.5 * loss_ret
            
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
        if args.output_dir:
            output_dir = args.output_dir
            if not os.path.exists(output_dir): os.makedirs(output_dir)
            model.save_pretrained(os.path.join(output_dir, f"lora_epoch{epoch}"))
            torch.save(hypernet.state_dict(), os.path.join(output_dir, f"hypernet_epoch{epoch}.pt"))
            print(f"Saved checkpoints to {output_dir}")
        else:
            model.save_pretrained(f"phase7_revised_lora_epoch{epoch+1}") # Increment epoch index for continuation
            torch.save(hypernet.state_dict(), f"phase7_revised_hypernet_epoch{epoch+1}.pt")
            print(f"Saved checkpoints for epoch {epoch+1}")

    print("Training complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default="phase7_cot_chunks.pt")
    parser.add_argument("--resume_lora", type=str, required=False, help="Path to resume LoRA")
    parser.add_argument("--resume_hypernet", type=str, required=False, help="Path to resume HyperNet")
    parser.add_argument("--output_dir", type=str, required=False)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=-1)
    args = parser.parse_args()
    train(args)
