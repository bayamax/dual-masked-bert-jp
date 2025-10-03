from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F

from .hf_model import load_tokenizer_and_model
from .optimize_virtual_token import build_position_ids_for_virtual


@torch.inference_mode()
def teacher_student_metrics(hf, text: str, follow_text: str, e_star: torch.Tensor, temperature: float, topk: int) -> dict:
    tok = hf.tokenizer
    # Teacher ids
    pref = tok(text, return_tensors="pt", add_special_tokens=True)
    prefix_ids = pref["input_ids"].to(hf.device)
    foll = tok(follow_text, return_tensors="pt", add_special_tokens=False)
    follow_ids = foll["input_ids"].to(hf.device)

    # Teacher logits slices
    input_ids = torch.cat([prefix_ids, follow_ids], dim=1)
    attn = torch.ones_like(input_ids)
    out_t = hf.model(input_ids=input_ids, attention_mask=attn)
    logits_t = out_t.logits  # [1, Ltot, V]
    P = prefix_ids.size(1)
    Lf = follow_ids.size(1)
    sl_t = logits_t[:, P - 1 : P - 1 + Lf, :]  # [1, Lf, V]
    logprobs_t = F.log_softmax(sl_t / temperature, dim=-1)

    # Student logits slices (e* + follow)
    emb_layer = hf.model.get_input_embeddings()
    follow_emb = emb_layer(follow_ids)
    e_star = e_star.to(device=follow_emb.device, dtype=follow_emb.dtype)  # [k, d] or [1, k, d]
    if e_star.dim() == 3:
        e_star = e_star.squeeze(0)
    virt = e_star.unsqueeze(0)  # [1, k, d]
    full_emb = torch.cat([virt, follow_emb], dim=1)  # [1, k+Lf, d]
    attn_s = torch.ones(full_emb.size()[:2], dtype=torch.long, device=full_emb.device)
    pos = build_position_ids_for_virtual(int(P), int(Lf), int(e_star.size(0))).to(hf.device)
    try:
        out_s = hf.model(inputs_embeds=full_emb, attention_mask=attn_s, position_ids=pos)
    except IndexError:
        out_s = hf.model(inputs_embeds=full_emb, attention_mask=attn_s)
    logits_s = out_s.logits
    sl_s = logits_s[:, e_star.size(0) - 1 : e_star.size(0) - 1 + Lf, :]
    logprobs_s = F.log_softmax(sl_s / temperature, dim=-1)

    # Metrics vs teacher
    probs_t = logprobs_t.exp()
    ce = (-(probs_t * logprobs_s).sum(dim=-1)).mean().item()

    # Top-1/Top-k of student vs ground-truth next token (teacher's follow tokens)
    true_ids = follow_ids  # [1, Lf]
    pred1 = logprobs_s.argmax(dim=-1)
    top1 = (pred1 == true_ids).float().mean().item()
    k = min(topk, logprobs_s.size(-1))
    vals, idx = torch.topk(logprobs_s, k=k, dim=-1)
    topk_acc = (idx == true_ids.unsqueeze(-1)).any(dim=-1).float().mean().item()

    # Teacher-only metrics vs ground-truth next token
    pred1_t = logprobs_t.argmax(dim=-1)
    top1_t = (pred1_t == true_ids).float().mean().item()
    vals_t, idx_t = torch.topk(logprobs_t, k=k, dim=-1)
    topk_t = (idx_t == true_ids.unsqueeze(-1)).any(dim=-1).float().mean().item()
    # Teacher CE on true tokens
    ce_t = (-(logprobs_t.gather(-1, true_ids.unsqueeze(-1)).squeeze(-1))).mean().item()

    return {
        "student_top1": top1,
        "student_topk": topk_acc,
        "student_ce_vs_teacher": ce,
        "teacher_top1": top1_t,
        "teacher_topk": topk_t,
        "teacher_ce_true": ce_t,
        "Lf": int(Lf),
    }


def iter_samples_from_stream(stream_dir: Path, head: int | None = None) -> Iterable[dict]:
    paths = sorted(glob.glob(str(stream_dir / "sample_*.pt")))
    if head is not None:
        paths = paths[: head]
    for p in paths:
        yield torch.load(p, map_location="cpu")


def iter_samples_from_agg(dataset_path: Path) -> Iterable[dict]:
    obj = torch.load(dataset_path, map_location="cpu")
    for s in obj["samples"]:
        yield s


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--dataset_path", type=Path, default=None)
    p.add_argument("--stream_dir", type=Path, default=None)
    p.add_argument("--head", type=int, default=None)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--temperature", type=float, default=1.0)
    args = p.parse_args()

    hf = load_tokenizer_and_model(args.model_name)

    if args.stream_dir is not None:
        samples = iter_samples_from_stream(args.stream_dir, head=args.head)
    elif args.dataset_path is not None:
        samples = iter_samples_from_agg(args.dataset_path)
    else:
        raise ValueError("either --dataset_path or --stream_dir is required")

    total = 0
    sum_st1 = sum_stk = sum_sce = 0.0
    sum_tt1 = sum_ttk = sum_tce = 0.0
    for i, s in enumerate(samples, 1):
        text = s.get("text", "")
        follow_text = s.get("follow_text", "")
        e_star = s["e_star"]  # [k, d]
        m = teacher_student_metrics(hf, text, follow_text, e_star, temperature=args.temperature, topk=args.topk)
        total += 1
        sum_st1 += m["student_top1"]
        sum_stk += m["student_topk"]
        sum_sce += m["student_ce_vs_teacher"]
        sum_tt1 += m["teacher_top1"]
        sum_ttk += m["teacher_topk"]
        sum_tce += m["teacher_ce_true"]
        if i % 20 == 0 or i == 1:
            print({"seen": i, "last": m})

    if total == 0:
        print({"total": 0})
        return
    avg_st1 = sum_st1 / total
    avg_stk = sum_stk / total
    avg_sce = sum_sce / total
    avg_tt1 = sum_tt1 / total
    avg_ttk = sum_ttk / total
    avg_tce = sum_tce / total
    print({
        "total": total,
        "student_top1": avg_st1,
        "teacher_top1": avg_tt1,
        "delta_top1": avg_st1 - avg_tt1,
        "student_topk": avg_stk,
        "teacher_topk": avg_ttk,
        "delta_topk": avg_stk - avg_ttk,
        "student_ce_vs_teacher": avg_sce,
        "teacher_ce_true": avg_tce,
        "topk": args.topk,
    })


if __name__ == "__main__":
    main()


