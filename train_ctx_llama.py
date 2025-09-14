#!/usr/bin/env python
"""Sentence-context vector reconstruction with Llama-3-8B-Instruct.
Assumes JSONL corpus: {"id": str, "sentences": ["...", ...]}
Run:
 accelerate launch train_ctx_llama.py --train data/train.jsonl --val data/val.jsonl
"""
from __future__ import annotations
import argparse, json, math, random, os
from pathlib import Path
from dataclasses import dataclass
from typing import List

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

# ---------------- utils -----------------

def set_seed(sd=42):
    random.seed(sd); torch.manual_seed(sd); torch.cuda.manual_seed_all(sd)

JP_SENT_END = "。"  # simple splitter

def split_sent(text: str) -> List[str]:
    return [s.strip() for s in text.split(JP_SENT_END) if s.strip()]

# -------------- data --------------------

class JsonlDocDS(Dataset):
    def __init__(self, path: str, max_sent: int = 50):
        self.items = []
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                obj = json.loads(ln)
                sents = obj.get("sentences")
                if not sents:
                    sents = split_sent(obj.get("text", ""))
                if len(sents) >= 2:
                    self.items.append(sents[:max_sent])
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        return self.items[idx]

@dataclass
class Batch:
    slots: torch.Tensor      # (B,K,d)
    labels: torch.Tensor     # (B,K,d)
    mask: torch.Tensor       # (B,K) 1=masked

# -------------- collate -----------------

def collate(batch: List[List[str]], tokenizer, model, K=8, p_mask=0.15, max_len=128):
    device = next(model.parameters()).device
    d = model.config.hidden_size
    slots = []; labels = []; mks = []
    for sents in batch:
        vecs = []
        for s in sents[:K]:
            ids = tokenizer(s, return_tensors="pt", truncation=True, max_length=max_len).input_ids.to(device)
            with torch.no_grad():
                out = model(input_ids=ids, use_cache=False, output_hidden_states=True)
            vecs.append(out.hidden_states[-1][0,-1])  # last token hidden
        if len(vecs) < K:
            pad = [torch.zeros_like(vecs[0])]*(K-len(vecs))
            vecs.extend(pad)
        vecs = torch.stack(vecs,0)  # (K,d)
        mask = torch.rand(K, device=device) < p_mask
        lbl = vecs.clone()
        vecs[mask] = 0.0
        slots.append(vecs); labels.append(lbl); mks.append(mask)
    return Batch(torch.stack(slots), torch.stack(labels), torch.stack(mks))

# -------------- loss --------------------

def vec_loss(pred, tgt, mask):
    mse = (pred - tgt).pow(2).sum(-1)
    cos = 1 - F.cosine_similarity(pred, tgt, dim=-1)
    loss = (mse + cos) * mask.float()
    return loss.sum() / mask.float().sum().clamp(min=1)

# -------------- main --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--bsz", type=int, default=2)
    args = ap.parse_args()
    set_seed()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        load_in_4bit=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    # --- Predictor MLP (d -> d) ---
    class VecPredictor(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Linear(hidden_size, 4*hidden_size, bias=False),
                nn.GELU(),
                nn.Linear(4*hidden_size, hidden_size, bias=False),
            )
        def forward(self, x):
            # x: (B,K,d)
            B,K,d = x.shape
            return self.mlp(x.view(B*K,d)).view(B,K,d)

    predictor = VecPredictor(base.config.hidden_size).to(args.device)

    # LoRA to predictor layers & base model (optional for memory)
    lora_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", target_modules=["q_proj","v_proj","linear"])
    base = get_peft_model(base, lora_cfg)
    predictor = get_peft_model(predictor, LoraConfig(r=8,lora_alpha=16,lora_dropout=0.05,bias="none",target_modules=["0","2"]))

    # combine parameters
    model = nn.ModuleDict({"llama": base, "pred": predictor})

    train_ds = JsonlDocDS(args.train)
    val_ds   = JsonlDocDS(args.val)
    train_ld = DataLoader(train_ds, batch_size=args.bsz, shuffle=True, collate_fn=lambda b: collate(b,tokenizer,model))
    val_ld   = DataLoader(val_ds,   batch_size=args.bsz, shuffle=False, collate_fn=lambda b: collate(b,tokenizer,model))

    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    model.train()
    for ep in range(1, args.epochs+1):
        tot=0;loss_sum=0
        for batch in tqdm(train_ld, desc=f"ep{ep}"):
            with torch.no_grad():
                slots = batch.slots  # (B,K,d) from llama hidden already
            pred = model["pred"](slots)
            loss = vec_loss(pred, batch.labels, batch.mask)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum+=loss.item(); tot+=1

        # ---- end epoch ----
        train_avg = loss_sum / max(1, tot)

        # validation
        model.eval(); vtot=0; vloss=0
        with torch.no_grad():
            for vb in val_ld:
                vp = model["pred"](vb.slots)
                l = vec_loss(vp, vb.labels, vb.mask)
                vloss += l.item(); vtot += 1
        val_avg = vloss / max(1, vtot)
        model.train()

        print(f"ep{ep} train {train_avg:.4f} | val {val_avg:.4f}")
    # save adapter
    model.save_pretrained("llama_ctx_adapter")
    print("saved LoRA adapter to llama_ctx_adapter/")

if __name__ == "__main__":
    main()
