
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig
from hyperprompt import HyperPromptNet
from hippocampus_v2 import HippocampusV2
from torch.utils.data import Dataset, DataLoader
import argparse
import os

class HippocampusDataset(Dataset):
    def __init__(self, hippo_dir):
        self.hippo = HippocampusV2(storage_dir=hippo_dir)
        self.hippo.load()
        # We iterate over indices that have labels (or all?)
        # Phase 1: Focus on items WITH labels ("Low Oxygen" needs "Oxygen")
        # Or should we train on everything?
        # User said "Use artificial breaks... maximize hippocampal muscle".
        # So we should train primarily on examples where we HAVE a target to retrieve.
        self.indices_with_labels = list(self.hippo.labels.keys())
        print(f"Dataset Size: {len(self.indices_with_labels)} labeled examples.")
        
    def __len__(self):
        return len(self.indices_with_labels)
        
    def __getitem__(self, idx):
        bank_idx = self.indices_with_labels[idx]
        target_idx = self.hippo.labels[bank_idx]
        
        current_text = self.hippo.get_text(bank_idx)
        target_text = self.hippo.get_text(target_idx)
        target_z = self.hippo.get_z(target_idx)
        
        return current_text, target_text, target_z

def collate_fn(batch):
    # Batch is list of (curr, targ_txt, targ_z)
    # Texts are tensors of variable length? No, chunk_size is fixed usually.
    # But just in case pad.
    
    currents, targets, zs = zip(*batch)
    
    # helper pad
    def pad(tensors):
        max_len = max(t.size(0) for t in tensors)
        padded = torch.zeros(len(tensors), max_len, dtype=torch.long)
        for i, t in enumerate(tensors):
            padded[i, :t.size(0)] = t
        return padded
        
    c_pad = pad(currents)
    t_pad = pad(targets)
    z_stack = torch.stack(zs) # [B, D]
    
    return c_pad, t_pad, z_stack

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, default="checkpoints_v2")
    parser.add_argument("--data_dir", type=str, default="hippocampus_v2_data")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=-1, help="Stop after N steps")
    args = parser.parse_args()
    
    os.makedirs(args.save_path, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Models
    print("Loading Models...")
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    
    # LoRA
    peft_config = LoraConfig(
        r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    student = get_peft_model(base_model, peft_config)
    student.print_trainable_parameters()
    
    # HyperNet
    hypernet = HyperPromptNet(d_model=2048, num_virtual=1, output_dim=512).to(device).float()
    
    # Optimizer
    opt = torch.optim.AdamW([
        {'params': student.parameters(), 'lr': 1e-4},
        {'params': hypernet.parameters(), 'lr': 1e-3}
    ])
    
    # DataLoader
    ds = HippocampusDataset(args.data_dir)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    
    # 2. Training Loop
    student.train()
    hypernet.train()
    
    print("Starting Learning Loop (The 2-Pass)...")
    
    for epoch in range(args.epochs):
        for step, (curr_ids, targ_ids, targ_z_gt) in enumerate(dl):
            curr_ids = curr_ids.to(device)
            targ_ids = targ_ids.to(device)
            targ_z_gt = targ_z_gt.to(device).float()
            
            # --- Pass 1: Speculative (Recall) ---
            # Forward Student on Current to get Hidden
            # Note: We want Hidden for the *First* token? Or *Last*?
            # To predict "What I need", we usually look at the start or end of context.
            # "Speculative Pass: Input window... Hypernet generates query"
            # Let's use the LAST token's hidden state as the query summary?
            # Or the FIRST? 
            # If we want to retrieve context *before* processing the window, we should use the first token?
            # But the first token has no context!
            # The window *is* the context. We want to retrieve *extended* context.
            # So we process the window, realize we are missing something, and query.
            # So Hidden at LAST token.
            
            outputs_1 = student(curr_ids, output_hidden_states=True)
            hidden_1 = outputs_1.hidden_states[-1] # [B, Seq, Dim]
            
            # Z-Query from Last Token (or Mean?)
            # Using Mean for robustness
            z_query = hypernet(hidden_1.float().mean(dim=1).unsqueeze(1)).squeeze(1) # [B, 512]
            
            # Loss 1: InfoNCE between z_query and targ_z_gt
            z_q_norm = F.normalize(z_query, p=2, dim=1)
            z_t_norm = F.normalize(targ_z_gt, p=2, dim=1)
            
            temp = 0.07
            logits_recall = torch.mm(z_q_norm, z_t_norm.t()) / temp
            labels_recall = torch.arange(logits_recall.size(0), device=device)
            loss_recall = F.cross_entropy(logits_recall, labels_recall)
            
            # --- Pass 2: Grounded (Generation) ---
            # Now we "Summon" the KV.
            # 1. Regenerate KV for Target
            # We want Target to appear *Before* Current.
            # Virtual Positions:
            # Target: [0, ..., T_len-1]
            # Current: [T_len, ..., T_len + C_len - 1]
            
            # Since batch items might have different lengths, careful with masking.
            # But here we pad.
            
            # Run Target through Base Model to get KV
            # We must use position_ids explicitly
            # Fix: Use actual batch size from input, not args.batch_size
            curr_batch_size = curr_ids.size(0)
            t_len = targ_ids.size(1)
            t_pos = torch.arange(0, t_len, device=device).unsqueeze(0).expand(curr_batch_size, -1)
            
            with torch.no_grad(): # Don't train on Target Processing? Or do we?
                # "KV cache... regenerated".
                # Usually we don't backprop through the "Memory Fetch" generation itself unless gradients flow through Z?
                # But here Z generation is detached from KV usage step selection-wise (hard decision).
                # So No Grad on KV gen is safe/efficient for memory?
                # Actually, LoRA applies to everything. 
                # If we want student to learn to *read* the retrieved KV, we need grad flow during Pass 2.
                # But do we update weights based on *how we processed the past*?
                # Yes.
                # But for memory efficiency, maybe `no_grad` context for the "Past" processing.
                outputs_past = student(targ_ids, position_ids=t_pos, use_cache=True)
                past_kv = outputs_past.past_key_values
                
            # 2. Run Current with Injection
            # Current Positions start after Target
            c_len = curr_ids.size(1)
            c_pos = torch.arange(t_len, t_len + c_len, device=device).unsqueeze(0).expand(curr_batch_size, -1)
            
            # Pass 2 Forward
            outputs_2 = student(curr_ids, position_ids=c_pos, past_key_values=past_kv, output_hidden_states=True)
            logits_2 = outputs_2.logits
            
            # Loss 2: Generation (CLM)
            # Shift labels
            shift_logits = logits_2[..., :-1, :].contiguous()
            shift_labels = curr_ids[..., 1:].contiguous()
            loss_gen = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # Total Loss
            loss = loss_recall + loss_gen
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            if step % 10 == 0:
                print(f"Step {step}: Recall={loss_recall.item():.4f}, Gen={loss_gen.item():.4f}")
                
            if args.max_steps > 0 and step >= args.max_steps:
                print("Max steps reached. Stopping.")
                break
        
        # Save per epoch (or end)
        student.save_pretrained(f"{args.save_path}/student_last")
        torch.save(hypernet.state_dict(), f"{args.save_path}/hypernet_last.pt")
        
        if args.max_steps > 0 and step >= args.max_steps:
            break

if __name__ == "__main__":
    main()
