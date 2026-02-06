"""
Phase 7 SFT Training Script

Trains TinyLlama to properly read [参照情報: ...] blocks and answer correctly.
Uses LoRA for efficient fine-tuning.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
import json
import argparse

# Configuration
STUDENT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATA_FILE = "phase7_sft_injection.jsonl"
MAX_LENGTH = 512
BATCH_SIZE = 4
LEARNING_RATE = 2e-4
EPOCHS = 3

class SFTDataset(Dataset):
    def __init__(self, data_file, tokenizer, max_length):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        with open(data_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                self.samples.append(data)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Combine input and output for causal LM training
        full_text = sample['input'] + sample['output']
        
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        # Labels: -100 for input part (don't compute loss), actual tokens for output
        labels = input_ids.clone()
        
        # Find where the output starts (after "Model:")
        input_len = len(self.tokenizer.encode(sample['input'], add_special_tokens=False))
        labels[:input_len] = -100  # Mask input tokens
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load tokenizer and model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Apply LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Load dataset
    dataset = SFTDataset(args.data_file, tokenizer, MAX_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print(f"Dataset size: {len(dataset)}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    model.train()
    
    for epoch in range(args.epochs):
        total_loss = 0
        for step, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if step % 10 == 0:
                print(f"Epoch {epoch} Step {step} Loss: {loss.item():.4f}")
            
            if args.max_steps > 0 and step >= args.max_steps:
                print("Max steps reached.")
                break
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        model.save_pretrained(f"phase7_sft_lora_epoch{epoch}")
        print(f"Saved checkpoint: phase7_sft_lora_epoch{epoch}")
    
    print("Training complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default=DATA_FILE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--max_steps", type=int, default=-1)
    args = parser.parse_args()
    train(args)
