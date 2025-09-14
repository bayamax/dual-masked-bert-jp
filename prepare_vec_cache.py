#!/usr/bin/env python
"""Precompute sentence vectors and save sharded cache.
Each shard file is a torch tensor of shape (N_shard, K, d) in bfloat16.
Usage:
 python prepare_vec_cache.py \
    --input data/train.jsonl \
    --output_prefix data/cache/train_vecs \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --K 8 --shard_size 10000 --max_docs 0
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

def split_sent(text: str):
    jp_end = "。"
    return [s.strip() for s in text.split(jp_end) if s.strip()]

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output_prefix", required=True, help="e.g. data/cache/train_vecs")
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--shard_size", type=int, default=10000)
    ap.add_argument("--max_docs", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    Path(os.path.dirname(args.output_prefix)).mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        load_in_4bit=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    d = model.config.hidden_size

    shard_idx = 0
    buf = []

    def flush():
        nonlocal shard_idx, buf
        if not buf:
            return
        tensor = torch.stack(buf)  # (N,K,d)
        out_path = f"{args.output_prefix}_{shard_idx:06d}.pt"
        torch.save(tensor.cpu(), out_path)
        print(f"saved {out_path} shape={tensor.shape}")
        shard_idx += 1
        buf = []

    with open(args.input, "r", encoding="utf-8") as f:
        for doc_i, line in enumerate(tqdm(f, desc="docs")):
            if args.max_docs and doc_i >= args.max_docs:
                break
            obj = json.loads(line)
            sents = obj.get("sentences") or split_sent(obj.get("text", ""))
            vecs = []
            for s in sents[: args.K]:
                ids = tok(s, return_tensors="pt", truncation=True, max_length=128).input_ids.to(args.device)
                out = model(input_ids=ids, use_cache=False, output_hidden_states=True)
                vecs.append(out.hidden_states[-1][0, -1].cpu())
            # pad
            while len(vecs) < args.K:
                vecs.append(torch.zeros(d, dtype=torch.bfloat16))
            buf.append(torch.stack(vecs).to(torch.bfloat16))
            if len(buf) >= args.shard_size:
                flush()
    flush()
    print("Finished all docs")

if __name__ == "__main__":
    main()
