"""
Phase 7 Unified Training

Trains HyperNet + LoRA together for:
1. Retrieval (InfoNCE contrastive loss)
2. Generation with reference format (English/DeepSeek style)

Format: <ref>{retrieved_context}</ref>\nQuestion: {question}\nAnswer: Based on the reference, the answer is {answer}.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
import json
import re
import argparse

# Configuration
STUDENT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
HYPERNET_DIM = 2048
CHUNK_SIZE = 127
MAX_LENGTH = 512
BATCH_SIZE = 8
LEARNING_RATE = 2e-4
TEMPERATURE = 0.07

class HyperNetHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.proj(x)

class UnifiedDataset(Dataset):
    """Dataset for unified retrieval + generation training."""
    
    def __init__(self, raw_file, tokenizer, max_length=MAX_LENGTH):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        with open(raw_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    question = data['question']
                    cot = data['generated_cot']
                    
                    # Extract answer
                    answer = self._extract_answer(cot)
                    if not answer:
                        continue
                    
                    # Create reference context (truncated CoT)
                    ref_context = cot[:300] if len(cot) > 300 else cot
                    
                    self.samples.append({
                        'question': question,
                        'ref_context': ref_context,
                        'answer': answer
                    })
                except:
                    continue
        
        print(f"Loaded {len(self.samples)} samples")
    
    def _extract_answer(self, cot_text):
        # Look for boxed answer
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', cot_text)
        if boxed_match:
            return boxed_match.group(1)
        
        # Look for "= X" pattern
        eq_match = re.search(r'=\s*(\d+)', cot_text)
        if eq_match:
            return eq_match.group(1)
        
        return None
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch):
    return batch

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load tokenizer and model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager"
    ).to(device)
    
    # Apply LoRA
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
    dataset = UnifiedDataset(args.data_file, tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    
    # Optimizer
    optimizer = torch.optim.AdamW([
        {'params': model.parameters()},
        {'params': hypernet.parameters()}
    ], lr=LEARNING_RATE)
    
    model.train()
    hypernet.train()
    
    print("Starting Unified Training...")
    
    for epoch in range(args.epochs):
        total_loss = 0
        total_ret_loss = 0
        total_gen_loss = 0
        
        for step, batch in enumerate(dataloader):
            bs = len(batch)
            
            questions = [s['question'] for s in batch]
            ref_contexts = [s['ref_context'] for s in batch]
            answers = [s['answer'] for s in batch]
            
            # ===== RETRIEVAL OBJECTIVE =====
            # Embed queries
            inputs_q = tokenizer(questions, return_tensors="pt", padding=True, truncation=True).to(device)
            outputs_q = model.get_base_model()(inputs_q.input_ids, attention_mask=inputs_q.attention_mask, output_hidden_states=True)
            last_hidden_q = outputs_q.hidden_states[-1][:, -1, :]
            z_query = hypernet(last_hidden_q.float())
            
            # Embed references (using recursive-style with dummy z_prev)
            inputs_ref = tokenizer(ref_contexts, return_tensors="pt", padding=True, truncation=True, max_length=CHUNK_SIZE).to(device)
            z_prev_dummy = torch.zeros(bs, 1, model.config.hidden_size, device=device).bfloat16()
            ref_embeds = model.get_base_model().model.embed_tokens(inputs_ref.input_ids)
            combined_embeds = torch.cat([z_prev_dummy, ref_embeds], dim=1)
            out_ref = model.get_base_model()(inputs_embeds=combined_embeds, output_hidden_states=True)
            last_hidden_ref = out_ref.hidden_states[-1][:, -1, :]
            z_ref = hypernet(last_hidden_ref.float())
            
            # InfoNCE Loss
            z_q_norm = F.normalize(z_query, p=2, dim=1)
            z_ref_norm = F.normalize(z_ref, p=2, dim=1)
            logits = torch.mm(z_q_norm, z_ref_norm.t()) / TEMPERATURE
            labels = torch.arange(bs, device=device)
            loss_retrieval = F.cross_entropy(logits, labels)
            
            # ===== GENERATION OBJECTIVE =====
            # Format: <ref>{context}</ref>\nQuestion: {q}\nAnswer: Based on the reference, the answer is {a}.
            gen_inputs = []
            gen_targets = []
            for i in range(bs):
                input_text = f"<ref>{ref_contexts[i]}</ref>\nQuestion: {questions[i]}\nAnswer:"
                target_text = f" Based on the reference, the answer is {answers[i]}."
                gen_inputs.append(input_text)
                gen_targets.append(target_text)
            
            # Tokenize for generation loss
            full_texts = [inp + tgt for inp, tgt in zip(gen_inputs, gen_targets)]
            inputs_gen = tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH).to(device)
            
            # Create labels (mask input part)
            labels_gen = inputs_gen.input_ids.clone()
            for i in range(bs):
                input_len = len(tokenizer.encode(gen_inputs[i], add_special_tokens=False))
                labels_gen[i, :input_len] = -100
            
            outputs_gen = model(
                inputs_gen.input_ids,
                attention_mask=inputs_gen.attention_mask,
                labels=labels_gen
            )
            loss_gen = outputs_gen.loss
            
            # ===== COMBINED LOSS =====
            loss = loss_retrieval + loss_gen
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_ret_loss += loss_retrieval.item()
            total_gen_loss += loss_gen.item()
            
            if step % 10 == 0:
                print(f"Epoch {epoch} Step {step} | Total: {loss.item():.4f} | Ret: {loss_retrieval.item():.4f} | Gen: {loss_gen.item():.4f}")
            
            if args.max_steps > 0 and step >= args.max_steps:
                print("Max steps reached.")
                break
        
        avg_loss = total_loss / len(dataloader)
        avg_ret = total_ret_loss / len(dataloader)
        avg_gen = total_gen_loss / len(dataloader)
        print(f"Epoch {epoch} | Avg Loss: {avg_loss:.4f} | Avg Ret: {avg_ret:.4f} | Avg Gen: {avg_gen:.4f}")
        
        # Save checkpoints
        model.save_pretrained(f"phase7_unified_lora_epoch{epoch}")
        torch.save(hypernet.state_dict(), f"phase7_unified_hypernet_epoch{epoch}.pt")
        print(f"Saved: phase7_unified_lora_epoch{epoch}, phase7_unified_hypernet_epoch{epoch}.pt")
    
    print("Training complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default="phase7_raw_cot.jsonl")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=-1)
    args = parser.parse_args()
    train(args)
