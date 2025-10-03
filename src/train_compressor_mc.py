from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hf_model import load_tokenizer_and_model


@dataclass
class TrainConfig:
    model_name: str
    num_virtual: int
    steps: int
    lr: float
    temperature: float
    device: str
    hidden_layer_index: int
    hidden_loss_type: str
    hidden_weight: float
    follow_len: int
    ctx_len: int
    batch_size: int
    random_embed_mode: str
    save_path: Path
    val_steps: int
    val_interval: int
    log_every: int


class QueryAttentionCompressor(nn.Module):
    """学習可能なk個のクエリで文脈隠れ状態にクロスアテンションし、k個のe*を出力"""

    def __init__(self, d_model: int, num_virtual: int = 1, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.num_virtual = num_virtual
        self.query = nn.Parameter(torch.randn(num_virtual, d_model))
        nn.init.xavier_uniform_(self.query)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.out = nn.Linear(d_model, d_model)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, context_states: torch.Tensor) -> torch.Tensor:
        # context_states: [B, Lc, d]
        B = context_states.size(0)
        queries = self.query.to(dtype=context_states.dtype, device=context_states.device)
        queries = queries.unsqueeze(0).expand(B, -1, -1)  # [B, k, d]
        attn_out, _ = self.attn(query=queries, key=context_states, value=context_states, need_weights=False)
        out = self.out(attn_out)
        return self.ln(out)  # [B, k, d]


@torch.no_grad()
def compute_avg_input_norm(hf) -> float:
    emb = hf.model.get_input_embeddings().weight.detach().to(torch.float32)
    return emb.norm(dim=-1).mean().item()


def sample_random_context_embeds(
    d_model: int,
    batch_size: int,
    ctx_len: int,
    device: torch.device,
    avg_norm: float,
    mode: str = "gauss_norm",
    emb_matrix: torch.Tensor | None = None,
    target_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if mode == "gauss_norm":
        x = torch.randn(batch_size, ctx_len, d_model, device=device)
        x = x / (x.norm(dim=-1, keepdim=True) + 1e-8) * avg_norm
        if target_dtype is not None:
            x = x.to(dtype=target_dtype)
        return x
    elif mode == "from_vocab":
        assert emb_matrix is not None, "emb_matrix is required for from_vocab mode"
        V = emb_matrix.size(0)
        ids = torch.randint(low=0, high=V, size=(batch_size, ctx_len), device=device)
        x = emb_matrix[ids].to(device=device)  # [B, Lc, d]
        # 微小ノイズで多様性を付与
        noise = (0.001 * avg_norm) * torch.randn_like(x)
        x = x + noise
        return x
    else:
        raise ValueError(f"unknown random_embed_mode: {mode}")


def compute_hidden_loss(pred: torch.Tensor, targ: torch.Tensor, typ: str) -> torch.Tensor:
    if typ == "mse":
        return F.mse_loss(pred, targ)
    if typ == "cosine":
        pred_n = F.normalize(pred, dim=-1)
        targ_n = F.normalize(targ, dim=-1)
        return (1.0 - (pred_n * targ_n).sum(dim=-1)).mean()
    if typ == "cosine_norm":
        pred_n = F.normalize(pred, dim=-1)
        targ_n = F.normalize(targ, dim=-1)
        cos = (pred_n * targ_n).sum(dim=-1)
        norm_l2 = (pred.norm(dim=-1) - targ.norm(dim=-1)).pow(2)
        return (1.0 - cos + 0.1 * norm_l2).mean()
    raise ValueError(f"unknown hidden_loss_type: {typ}")


def train(cfg: TrainConfig) -> None:
    hf = load_tokenizer_and_model(cfg.model_name)
    device = torch.device(cfg.device if cfg.device != "auto" else hf.device.type)
    if device.type == "cpu":
        hf.model.to(dtype=torch.float32, device=device)
    else:
        hf.model.to(device)
    hf.model.requires_grad_(False)
    hf.model.eval()

    emb_layer = hf.model.get_input_embeddings()
    d_model = emb_layer.weight.shape[1]
    avg_norm = compute_avg_input_norm(hf)
    param_dtype = emb_layer.weight.dtype

    compressor = QueryAttentionCompressor(d_model=d_model, num_virtual=cfg.num_virtual, num_heads=8)
    # 圧縮器のdtypeをLLMの埋め込みdtypeに合わせる（CPU/MPSでの不一致回避）
    compressor = compressor.to(device=device, dtype=param_dtype)
    opt = torch.optim.AdamW(compressor.parameters(), lr=cfg.lr, weight_decay=1e-4)

    vocab_size = hf.tokenizer.vocab_size

    best_val: float | None = None

    def run_batch(device_batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        emb_weight = emb_layer.weight  # dtype基準
        ctx_emb = sample_random_context_embeds(
            d_model=d_model,
            batch_size=device_batch_size,
            ctx_len=cfg.ctx_len,
            device=device,
            avg_norm=avg_norm,
            mode=cfg.random_embed_mode,
            emb_matrix=emb_weight if cfg.random_embed_mode == "from_vocab" else None,
            target_dtype=emb_weight.dtype,
        )  # [B, Lc, d]

        follow_ids = torch.randint(low=0, high=vocab_size, size=(device_batch_size, cfg.follow_len), device=device)
        follow_emb = emb_layer(follow_ids)  # [B, Lf, d]

        with torch.no_grad():
            # dtype整合
            ctx_cast = ctx_emb.to(dtype=follow_emb.dtype)
            full_t = torch.cat([ctx_cast, follow_emb], dim=1)
            attn_t = torch.ones(full_t.size()[:2], dtype=torch.long, device=device)
            out_t = hf.model(inputs_embeds=full_t, attention_mask=attn_t, use_cache=False, output_hidden_states=True)
            logits_t = out_t.logits
            h_t = out_t.hidden_states[cfg.hidden_layer_index]
            sl_t = logits_t[:, cfg.ctx_len - 1 : cfg.ctx_len - 1 + cfg.follow_len, :]
            hs_t = h_t[:, cfg.ctx_len - 1 : cfg.ctx_len - 1 + cfg.follow_len, :]
            probs_t = F.log_softmax(sl_t / cfg.temperature, dim=-1).exp()

        # 文脈の隠れ状態シーケンスをそのまま入力
        ctx_states = h_t[:, : cfg.ctx_len, :]
        e_star = compressor(ctx_states)
        e_star = e_star / (e_star.norm(dim=-1, keepdim=True) + 1e-8) * avg_norm
        e_star = e_star.to(dtype=follow_emb.dtype)

        full_s = torch.cat([e_star, follow_emb], dim=1)
        attn_s = torch.ones(full_s.size()[:2], dtype=torch.long, device=device)
        out_s = hf.model(inputs_embeds=full_s, attention_mask=attn_s, use_cache=False, output_hidden_states=True)
        logits_s = out_s.logits
        h_s = out_s.hidden_states[cfg.hidden_layer_index]
        sl_s = logits_s[:, cfg.num_virtual - 1 : cfg.num_virtual - 1 + cfg.follow_len, :]
        hs_s = h_s[:, cfg.num_virtual - 1 : cfg.num_virtual - 1 + cfg.follow_len, :]
        return probs_t, sl_s, hs_s, hs_t, e_star

    def evaluate() -> dict:
        compressor.eval()
        losses = []
        with torch.no_grad():
            for _ in range(max(1, cfg.val_steps)):
                probs_t, sl_s, hs_s, hs_t, _ = run_batch(device_batch_size=cfg.batch_size)
                logprobs_s = F.log_softmax(sl_s / cfg.temperature, dim=-1)
                ce = -(probs_t * logprobs_s).sum(dim=-1).mean()
                h_loss = compute_hidden_loss(hs_s, hs_t, cfg.hidden_loss_type)
                loss = ce + cfg.hidden_weight * h_loss
                losses.append(loss.item())
        compressor.train()
        return {"val_loss": float(sum(losses) / len(losses))}

    compressor.train()
    for step in range(1, cfg.steps + 1):
        # ランダム文脈（埋め込み）
        probs_t, sl_s, hs_s, hs_t, e_star = run_batch(device_batch_size=cfg.batch_size)

        # 損失: KL(教師||生徒) + hidden一致
        logprobs_s = F.log_softmax(sl_s / cfg.temperature, dim=-1)
        ce = -(probs_t * logprobs_s).sum(dim=-1).mean()
        h_loss = compute_hidden_loss(hs_s, hs_t, cfg.hidden_loss_type)
        loss = ce + cfg.hidden_weight * h_loss

        opt.zero_grad()
        loss.backward()
        # 勾配クリップでNaN/発散抑制
        torch.nn.utils.clip_grad_norm_(compressor.parameters(), max_norm=1.0)
        opt.step()

        if step % max(1, cfg.log_every) == 0 or step == 1:
            print({"step": step, "loss": float(loss.item()), "ce": float(ce.item()), "h_loss": float(h_loss.item())})

        # validation & best-save
        if (step % max(1, cfg.val_interval) == 0) or (step == cfg.steps):
            val = evaluate()
            val_loss = val["val_loss"]
            print({"step": step, **val, "note": "validation"})
            if (best_val is None) or (val_loss < best_val):
                best_val = val_loss
                cfg.save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "state_dict": compressor.state_dict(),
                    "meta": {
                        **asdict(cfg),
                        "d_model": int(d_model),
                        "best_val": float(best_val),
                    },
                }, cfg.save_path)
                print("saved best to:", cfg.save_path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--num_virtual", type=int, default=1)
    p.add_argument("--steps", type=int, default=80)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--device", type=str, default="auto", choices=["cpu", "mps", "auto"])
    p.add_argument("--hidden_layer_index", type=int, default=-2)
    p.add_argument("--hidden_loss_type", type=str, default="cosine", choices=["mse", "cosine", "cosine_norm"])
    p.add_argument("--hidden_weight", type=float, default=0.3)
    p.add_argument("--follow_len", type=int, default=32)
    p.add_argument("--ctx_len", type=int, default=48)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--random_embed_mode", type=str, default="gauss_norm", choices=["gauss_norm", "from_vocab"]) 
    p.add_argument("--save_path", type=Path, default=Path("artifacts/compressor.pt"))
    p.add_argument("--val_steps", type=int, default=32)
    p.add_argument("--val_interval", type=int, default=10)
    p.add_argument("--log_every", type=int, default=1)
    args = p.parse_args()

    cfg = TrainConfig(
        model_name=args.model_name,
        num_virtual=args.num_virtual,
        steps=args.steps,
        lr=args.lr,
        temperature=args.temperature,
        device=args.device,
        hidden_layer_index=args.hidden_layer_index,
        hidden_loss_type=args.hidden_loss_type,
        hidden_weight=args.hidden_weight,
        follow_len=args.follow_len,
        ctx_len=args.ctx_len,
        batch_size=args.batch_size,
        random_embed_mode=args.random_embed_mode,
        save_path=args.save_path,
        val_steps=args.val_steps,
        val_interval=args.val_interval,
        log_every=args.log_every,
    )
    train(cfg)


if __name__ == "__main__":
    main()


