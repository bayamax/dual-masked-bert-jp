from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from .hf_model import load_tokenizer_and_model
from .optimize_virtual_token import build_position_ids_for_virtual


@torch.inference_mode()
def get_teacher_slices(hf, prefix_ids: torch.Tensor, follow_ids: torch.Tensor, temperature: float) -> torch.Tensor:
    input_ids = torch.cat([prefix_ids, follow_ids], dim=1)
    attn = torch.ones_like(input_ids)
    out = hf.model(input_ids=input_ids, attention_mask=attn, use_cache=False)
    logits = out.logits  # [1, Ltot, V]
    P = prefix_ids.size(1)
    Lf = follow_ids.size(1)
    sl = logits[:, P - 1 : P - 1 + Lf, :]  # [1, Lf, V]
    return sl / temperature


@torch.inference_mode()
def get_student_slices(
    hf,
    e_star: torch.Tensor,
    follow_ids: torch.Tensor,
    prefix_len: int,
    temperature: float,
) -> torch.Tensor:
    emb_layer = hf.model.get_input_embeddings()
    follow_emb = emb_layer(follow_ids)
    virt = e_star.to(device=follow_emb.device, dtype=follow_emb.dtype).unsqueeze(0)  # [1, k, d]
    full_emb = torch.cat([virt, follow_emb], dim=1)  # [1, k+Lf, d]

    attn = torch.ones(full_emb.size()[:2], dtype=torch.long, device=full_emb.device)
    k = e_star.size(0)
    Lf = follow_ids.size(1)
    pos = build_position_ids_for_virtual(prefix_len, int(Lf), int(k)).to(full_emb.device)
    try:
        out = hf.model(inputs_embeds=full_emb, attention_mask=attn, position_ids=pos, use_cache=False)
    except IndexError:
        out = hf.model(inputs_embeds=full_emb, attention_mask=attn, use_cache=False)
    logits = out.logits  # [1, k+Lf, V]
    sl = logits[:, k - 1 : k - 1 + Lf, :]  # [1, Lf, V]
    return sl / temperature


@torch.inference_mode()
def accuracy_from_slices(logit_slices: torch.Tensor, true_ids: torch.Tensor, topk: int) -> dict:
    # logit_slices: [1, L, V], true_ids: [1, L]
    probs = F.log_softmax(logit_slices, dim=-1)
    L = true_ids.size(1)
    # top-1
    pred1 = probs.argmax(dim=-1)  # [1, L]
    top1 = (pred1 == true_ids).float().mean().item()
    # top-k
    k = min(topk, probs.size(-1))
    vals, idx = torch.topk(probs, k=k, dim=-1)
    # [1, L, k] vs [1, L, 1]
    matchk = (idx == true_ids.unsqueeze(-1)).any(dim=-1).float().mean().item()
    # CE over true token
    ce = (-probs.gather(-1, true_ids.unsqueeze(-1)).squeeze(-1)).mean().item()
    return {"top1": top1, "topk": matchk, "ce": ce, "k": k, "L": int(L)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--e_star_path", type=Path, required=True)
    p.add_argument("--text", type=str, default=None)
    p.add_argument("--follow_text", type=str, default=None)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--temperature", type=float, default=1.0)
    args = p.parse_args()

    hf = load_tokenizer_and_model(args.model_name)
    tok = hf.tokenizer

    obj = torch.load(args.e_star_path, map_location="cpu")
    e_star = obj["e_star"]  # [k, d]
    meta = obj.get("meta", {})
    text = args.text if args.text is not None else obj.get("text", "")
    follow_text = args.follow_text if args.follow_text is not None else obj.get("follow_text", "")

    pref = tok(text, return_tensors="pt", add_special_tokens=True)
    prefix_ids = pref["input_ids"].to(hf.device)
    foll = tok(follow_text, return_tensors="pt", add_special_tokens=False)
    follow_ids = foll["input_ids"].to(hf.device)

    # Teacher
    sl_t = get_teacher_slices(hf, prefix_ids, follow_ids, temperature=args.temperature)
    acc_t = accuracy_from_slices(sl_t, follow_ids, args.topk)

    # Student
    P = int(meta.get("prefix_len", prefix_ids.size(1)))
    sl_s = get_student_slices(hf, e_star, follow_ids, P, temperature=args.temperature)
    acc_s = accuracy_from_slices(sl_s, follow_ids, args.topk)

    print({
        "teacher": acc_t,
        "student": acc_s,
        "prefix_len": int(prefix_ids.size(1)),
        "follow_len": int(follow_ids.size(1)),
        "num_virtual": int(e_star.size(0)),
    })


if __name__ == "__main__":
    main()


