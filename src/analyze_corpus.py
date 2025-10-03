from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Iterable, Dict, List

import torch
import torch.nn.functional as F

from .hf_model import load_tokenizer_and_model
from .optimize_virtual_token import build_position_ids_for_virtual


def iter_samples_from_stream(stream_dir: Path, head: int | None = None) -> Iterable[dict]:
    paths = sorted(glob.glob(str(stream_dir / "sample_*.pt")))
    if head is not None:
        paths = paths[: head]
    for p in paths:
        yield torch.load(p, map_location="cpu")


def iter_samples_from_agg(dataset_path: Path, head: int | None = None) -> Iterable[dict]:
    obj = torch.load(dataset_path, map_location="cpu")
    samples: List[Dict] = obj.get("samples", [])
    if head is not None:
        samples = samples[: head]
    for s in samples:
        yield s


@torch.inference_mode()
def teacher_slices(hf, text: str, follow_text: str, temperature: float):
    tok = hf.tokenizer
    pref = tok(text, return_tensors="pt", add_special_tokens=True)
    foll = tok(follow_text, return_tensors="pt", add_special_tokens=False)
    prefix_ids = pref["input_ids"].to(hf.device)
    follow_ids = foll["input_ids"].to(hf.device)
    input_ids = torch.cat([prefix_ids, follow_ids], dim=1)
    attn = torch.ones_like(input_ids)
    out = hf.model(input_ids=input_ids, attention_mask=attn, use_cache=False)
    logits = out.logits
    P = prefix_ids.size(1)
    Lf = follow_ids.size(1)
    sl = logits[:, P - 1 : P - 1 + Lf, :]
    logprobs = F.log_softmax(sl / temperature, dim=-1)
    return logprobs, follow_ids, int(P), int(Lf)


@torch.inference_mode()
def student_slices_with_estar(hf, e_star: torch.Tensor, follow_ids: torch.Tensor, prefix_len: int, temperature: float):
    emb_layer = hf.model.get_input_embeddings()
    follow_emb = emb_layer(follow_ids)
    e_star = e_star.to(device=follow_emb.device, dtype=follow_emb.dtype)
    if e_star.dim() == 2:
        virt = e_star.unsqueeze(0)  # [1,k,d]
    else:
        virt = e_star
    full = torch.cat([virt, follow_emb], dim=1)
    attn = torch.ones(full.size()[:2], dtype=torch.long, device=full.device)
    k = virt.size(1)
    Lf = follow_ids.size(1)
    pos = build_position_ids_for_virtual(int(prefix_len), int(Lf), int(k)).to(full.device)
    try:
        out = hf.model(inputs_embeds=full, attention_mask=attn, position_ids=pos, use_cache=False)
    except Exception:
        out = hf.model(inputs_embeds=full, attention_mask=attn, use_cache=False)
    logits = out.logits
    sl = logits[:, k - 1 : k - 1 + Lf, :]
    logprobs = F.log_softmax(sl / temperature, dim=-1)
    return logprobs


@torch.inference_mode()
def student_slices_with_compressor(hf, compressor, Hctx: torch.Tensor, follow_ids: torch.Tensor, prefix_len: int, avg_norm: torch.Tensor, temperature: float):
    H = Hctx.unsqueeze(0).to(device=hf.device, dtype=hf.dtype)
    e_pred = compressor(H, None)  # [1,k,d]
    e_pred = e_pred / (e_pred.norm(dim=-1, keepdim=True) + 1e-8) * avg_norm
    emb_layer = hf.model.get_input_embeddings()
    follow_emb = emb_layer(follow_ids)
    full = torch.cat([e_pred, follow_emb], dim=1)
    attn = torch.ones(full.size()[:2], dtype=torch.long, device=full.device)
    k = e_pred.size(1)
    Lf = follow_ids.size(1)
    pos = build_position_ids_for_virtual(int(prefix_len), int(Lf), int(k)).to(full.device)
    try:
        out = hf.model(inputs_embeds=full, attention_mask=attn, position_ids=pos, use_cache=False)
    except Exception:
        out = hf.model(inputs_embeds=full, attention_mask=attn, use_cache=False)
    logits = out.logits
    sl = logits[:, k - 1 : k - 1 + Lf, :]
    logprobs = F.log_softmax(sl / temperature, dim=-1)
    return logprobs, e_pred.squeeze(0)


def summarize(vals: torch.Tensor) -> Dict[str, float]:
    if vals.numel() == 0:
        return {"count": 0}
    v = vals.float().cpu()
    return {
        "count": int(v.numel()),
        "mean": float(v.mean().item()),
        "p50": float(v.median().item()),
        "p90": float(v.kthvalue(int(max(1, round(0.9 * v.numel()))))[0].item()) if v.numel() > 1 else float(v.item()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--dataset_path", type=Path, default=None)
    ap.add_argument("--stream_dir", type=Path, default=None)
    ap.add_argument("--head", type=int, default=None)
    ap.add_argument("--compressor_path", type=Path, default=None)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=1.0)
    args = ap.parse_args()

    hf = load_tokenizer_and_model(args.model_name)
    tok = hf.tokenizer

    # Load compressor if provided
    compressor = None
    avg_norm = None
    if args.compressor_path is not None and args.compressor_path.exists():
        ckpt = torch.load(args.compressor_path, map_location="cpu")
        meta = ckpt.get("meta", {})
        d_model = int(meta.get("d_model", hf.model.get_input_embeddings().weight.shape[1]))
        num_virtual = int(meta.get("num_virtual", 1))
        from .demo_compressor_sup import QueryAttentionCompressor  # lightweight local def
        from .hyperprompt import HyperPromptNet
        if str(meta.get("arch", "hyper")).lower() == "hyper":
            mdl = HyperPromptNet(
                d_model=d_model,
                num_virtual=num_virtual,
                num_heads=int(meta.get("num_heads", 8)),
                num_layers=int(meta.get("hyper_num_layers", 2)),
                ffn_dim=int(meta.get("hyper_ffn_dim", 2048)),
                dropout=float(meta.get("dropout", 0.1)),
                prenorm=bool(meta.get("hyper_prenorm", True)),
            )
        else:
            mdl = QueryAttentionCompressor(d_model=d_model, num_virtual=num_virtual, num_heads=8)
        mdl.load_state_dict(ckpt["state_dict"], strict=False)
        mdl.to(device=hf.device, dtype=hf.dtype).eval()
        compressor = mdl
        with torch.no_grad():
            avg_norm = hf.model.get_input_embeddings().weight.detach().to(torch.float32).norm(dim=-1).mean().to(hf.device)

    # Choose iterator
    if args.stream_dir is not None:
        samples = iter_samples_from_stream(args.stream_dir, head=args.head)
    elif args.dataset_path is not None:
        samples = iter_samples_from_agg(args.dataset_path, head=args.head)
    else:
        raise ValueError("either --dataset_path or --stream_dir is required")

    # Accumulators
    ce_teacher_true: List[float] = []
    top1_teacher_true: List[float] = []
    topk_teacher_true: List[float] = []

    ce_student_vs_teacher: List[float] = []
    top1_student_true: List[float] = []
    topk_student_true: List[float] = []

    cos_vec: List[float] = []

    for s in samples:
        text = s.get("text", "")
        follow = s.get("follow_text", "")
        if not text or not follow:
            continue
        logp_t, follow_ids, P, Lf = teacher_slices(hf, text, follow, temperature=args.temperature)
        probs_t = logp_t.exp()
        # CE on true next tokens
        ce_t = (-(logp_t.gather(-1, follow_ids.unsqueeze(-1)).squeeze(-1))).mean().item()
        ce_teacher_true.append(ce_t)
        # Top-1 / Top-k of teacher vs true
        pred1_t = logp_t.argmax(dim=-1)
        top1_teacher_true.append(float((pred1_t == follow_ids).float().mean().item()))
        k = min(args.topk, logp_t.size(-1))
        _, idx_t = torch.topk(logp_t, k=k, dim=-1)
        topk_teacher_true.append(float((idx_t == follow_ids.unsqueeze(-1)).any(dim=-1).float().mean().item()))

        # Student
        if compressor is not None:
            Hctx = s.get("Hctx")
            if isinstance(Hctx, torch.Tensor):
                logp_s, e_pred = student_slices_with_compressor(hf, compressor, Hctx, follow_ids, P, avg_norm, args.temperature)
                # vec cos vs stored e*（device/dtypeを揃え、k不一致ならminで比較）
                e_star = s.get("e_star")
                if isinstance(e_star, torch.Tensor):
                    e_star = e_star.to(device=e_pred.device, dtype=e_pred.dtype)
                    k_min = min(e_pred.size(0), e_star.size(0)) if e_pred.dim() == 2 and e_star.dim() == 2 else None
                    if k_min is not None and k_min > 0:
                        v1 = F.normalize(e_pred[:k_min].reshape(-1).float(), dim=0)
                        v2 = F.normalize(e_star[:k_min].reshape(-1).float(), dim=0)
                        cos = torch.clamp((v1 * v2).sum(), -1.0, 1.0).item()
                        cos_vec.append(float(cos))
        else:
            e_star = s.get("e_star")
            if isinstance(e_star, torch.Tensor):
                logp_s = student_slices_with_estar(hf, e_star, follow_ids, P, args.temperature)
            else:
                continue

        # CE(probs_t, logp_s) and student vs true
        ce_s = (-(probs_t * logp_s).sum(dim=-1)).mean().item()
        ce_student_vs_teacher.append(ce_s)
        pred1_s = logp_s.argmax(dim=-1)
        top1_student_true.append(float((pred1_s == follow_ids).float().mean().item()))
        _, idx_s = torch.topk(logp_s, k=k, dim=-1)
        topk_student_true.append(float((idx_s == follow_ids.unsqueeze(-1)).any(dim=-1).float().mean().item()))

    def t(x: List[float]):
        return summarize(torch.tensor(x)) if x else {"count": 0}

    print({
        "teacher_ce_true": t(ce_teacher_true),
        "teacher_top1_true": t(top1_teacher_true),
        "teacher_topk_true": t(topk_teacher_true),
        "student_ce_vs_teacher": t(ce_student_vs_teacher),
        "student_top1_true": t(top1_student_true),
        "student_topk_true": t(topk_student_true),
        "vec_cos": t(cos_vec),
    })


if __name__ == "__main__":
    main()


