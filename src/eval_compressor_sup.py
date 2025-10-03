from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .hf_model import load_tokenizer_and_model
from .optimize_virtual_token import build_position_ids_for_virtual
from .hyperprompt import HyperPromptNet


class QueryAttentionCompressor(nn.Module):
    def __init__(self, d_model: int, num_virtual: int = 1, num_heads: int = 8):
        super().__init__()
        self.num_virtual = num_virtual
        self.query = nn.Parameter(torch.randn(num_virtual, d_model))
        nn.init.xavier_uniform_(self.query)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.out = nn.Sequential(nn.Linear(d_model, d_model), nn.LayerNorm(d_model))

    def forward(self, ctx_states: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        b, l, d = ctx_states.size()
        q = self.query.unsqueeze(0).expand(b, -1, -1).to(dtype=ctx_states.dtype, device=ctx_states.device)
        kpm = None
        if attn_mask is not None:
            kpm = (attn_mask == 0)
        attn_out, _ = self.attn(query=q, key=ctx_states, value=ctx_states, key_padding_mask=kpm, need_weights=False)
        return self.out(attn_out)  # [B, k, d]


def load_dataset(path: Path) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str], List[str]]:
    obj = torch.load(path, map_location="cpu")
    samples = obj["samples"]
    H = torch.stack([s["Hctx"].to(torch.float32) for s in samples], dim=0)
    M = torch.stack([s["Hmask"].to(torch.long) for s in samples], dim=0)
    E = torch.stack([s["e_star"].to(torch.float32) for s in samples], dim=0)
    texts: List[str] = [s.get("text", "") for s in samples]
    follows: List[str] = [s.get("follow_text", "") for s in samples]
    return H, M, E, texts, follows


class SupDataset(Dataset):
    def __init__(self, H: torch.Tensor, M: torch.Tensor, E: torch.Tensor, texts: List[str], follows: List[str]):
        self.H = H
        self.M = M
        self.E = E
        self.texts = texts
        self.follows = follows

    def __len__(self) -> int:
        return self.H.size(0)

    def __getitem__(self, idx: int):
        return self.H[idx], self.M[idx], self.E[idx], self.texts[idx], self.follows[idx]


@torch.no_grad()
def compute_avg_input_norm(hf) -> torch.Tensor:
    return hf.model.get_input_embeddings().weight.detach().to(torch.float32).norm(dim=-1).mean()


def evaluate(
    dataset_path: Path,
    compressor_path: Path,
    model_name: str,
    batch_size: int,
    device_str: str,
    requested_dtype: str,
    topk: int,
    temperature: float,
) -> None:
    # Load dataset
    H, M, E, texts, follows = load_dataset(dataset_path)
    N, L, d = H.size()

    dev = None if device_str == "auto" else torch.device(device_str)
    hf = load_tokenizer_and_model(model_name, requested_dtype=requested_dtype, device=dev)
    device = hf.device
    dtype = hf.dtype
    tok = hf.tokenizer
    hf.model.eval()
    hf.model.requires_grad_(False)

    avg_norm = compute_avg_input_norm(hf).to(device)

    # Load compressor
    ckpt = torch.load(compressor_path, map_location="cpu")
    meta = ckpt.get("meta", {}) if isinstance(ckpt, dict) else {}
    num_virtual = int(meta.get("num_virtual", 1))
    arch = str(meta.get("arch", "hyper")).lower()

    if arch == "hyper":
        model = HyperPromptNet(
            d_model=d,
            num_virtual=num_virtual,
            num_heads=int(meta.get("num_heads", 8)),
            num_layers=int(meta.get("hyper_num_layers", 2)),
            ffn_dim=int(meta.get("hyper_ffn_dim", 2048)),
            dropout=float(meta.get("dropout", 0.1)),
            prenorm=bool(meta.get("hyper_prenorm", True)),
        )
    else:
        model = QueryAttentionCompressor(d_model=d, num_virtual=num_virtual, num_heads=int(meta.get("num_heads", 8)))
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state)
    model.to(device=device, dtype=dtype)
    model.eval()

    ds = SupDataset(H, M, E, texts, follows)

    def collate(batch):
        Hs, Ms, Es, Ts, Fs = zip(*batch)
        return torch.stack(Hs, 0), torch.stack(Ms, 0), torch.stack(Es, 0), list(Ts), list(Fs)

    loader = DataLoader(ds, batch_size=min(batch_size, N), shuffle=False, collate_fn=collate)

    # Metrics accumulators
    sum_mse = 0.0
    sum_cos = 0.0
    sum_norm_rmse = 0.0
    sum_ce = 0.0
    sum_top1 = 0.0
    sum_topk = 0.0
    seen = 0

    for Hb, Mb, Eb, Tb, Fb in loader:
        B = Hb.size(0)
        Hb = Hb.to(device=device, dtype=dtype)
        Mb = Mb.to(device)
        Eb = Eb.to(device=device, dtype=dtype)

        # Predict e*
        pred = model(Hb, Mb)  # [B, k, d]
        pred_normed = pred / (pred.norm(dim=-1, keepdim=True) + 1e-8) * avg_norm

        # Vector regression metrics（k不一致は min(k) で比較）
        k_min = int(min(pred_normed.size(1), Eb.size(1)))
        pn = pred_normed[:, :k_min, :]
        eb = Eb[:, :k_min, :]
        mse = F.mse_loss(pn, eb).item()
        cos = F.cosine_similarity(
            F.normalize(pn.flatten(0, 1).float(), dim=-1),
            F.normalize(eb.flatten(0, 1).float(), dim=-1),
            dim=-1,
        ).mean().item()
        rmse_norm = torch.sqrt(torch.mean((pred_normed.norm(dim=-1) - Eb.norm(dim=-1)) ** 2)).item()

        # Teacher logits
        enc_t = tok(list(Tb), return_tensors="pt", padding=True, truncation=True, add_special_tokens=True)
        enc_f = tok(list(Fb), return_tensors="pt", padding=True, truncation=True, add_special_tokens=False)
        prefix_ids = enc_t["input_ids"].to(device)
        follow_ids = enc_f["input_ids"].to(device)
        input_ids = torch.cat([prefix_ids, follow_ids], dim=1)
        attn = torch.ones_like(input_ids)
        with torch.no_grad():
            out_t = hf.model(input_ids=input_ids, attention_mask=attn, use_cache=False)
        logits_t = out_t.logits
        P = prefix_ids.size(1)
        Lf = follow_ids.size(1)
        sl_t = logits_t[:, P - 1 : P - 1 + Lf, :]
        logprobs_t = F.log_softmax(sl_t / temperature, dim=-1)
        probs_t = logprobs_t.exp()

        # Student logits
        emb_layer = hf.model.get_input_embeddings()
        follow_emb = emb_layer(follow_ids)
        full_emb = torch.cat([pred_normed, follow_emb], dim=1)
        attn_s = torch.ones(full_emb.size()[:2], dtype=torch.long, device=device)
        try:
            k = int(pred_normed.size(1))
            pos_ids = build_position_ids_for_virtual(int(P), int(Lf), k).to(device)
            if pos_ids.size(0) == 1 and full_emb.size(0) > 1:
                pos_ids = pos_ids.expand(full_emb.size(0), -1)
            out_s = hf.model(inputs_embeds=full_emb, attention_mask=attn_s, position_ids=pos_ids, use_cache=False)
        except Exception:
            out_s = hf.model(inputs_embeds=full_emb, attention_mask=attn_s, use_cache=False)
        logits_s = out_s.logits
        sl_s = logits_s[:, (pred_normed.size(1) - 1) : (pred_normed.size(1) - 1 + Lf), :]
        logprobs_s = F.log_softmax(sl_s / temperature, dim=-1)

        # CE and accuracy vs ground-truth next tokens
        ce = (-(probs_t * logprobs_s).sum(dim=-1)).mean().item()
        pred1 = logprobs_s.argmax(dim=-1)
        top1 = (pred1 == follow_ids).float().mean().item()
        K = min(topk, logprobs_s.size(-1))
        _, idx = torch.topk(logprobs_s, k=K, dim=-1)
        topk_acc = (idx == follow_ids.unsqueeze(-1)).any(dim=-1).float().mean().item()

        # Accumulate
        seen += B
        sum_mse += mse * B
        sum_cos += cos * B
        sum_norm_rmse += rmse_norm * B
        sum_ce += ce * B
        sum_top1 += top1 * B
        sum_topk += topk_acc * B

    if seen == 0:
        print({"total": 0})
        return

    print({
        "total": int(seen),
        "vec_mse": float(sum_mse / seen),
        "vec_cos": float(sum_cos / seen),
        "vec_rmse_norm": float(sum_norm_rmse / seen),
        "student_ce_vs_teacher": float(sum_ce / seen),
        "student_top1": float(sum_top1 / seen),
        "student_topk": float(sum_topk / seen),
        "topk": int(topk),
        "temperature": float(temperature),
    })


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", type=Path, required=True)
    p.add_argument("--compressor_path", type=Path, required=True)
    p.add_argument("--model_name", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "auto"])
    p.add_argument("--dtype", type=str, default="float32", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--temperature", type=float, default=1.0)
    args = p.parse_args()

    evaluate(
        dataset_path=args.dataset_path,
        compressor_path=args.compressor_path,
        model_name=args.model_name,
        batch_size=args.batch_size,
        device_str=args.device,
        requested_dtype=args.dtype,
        topk=args.topk,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()


