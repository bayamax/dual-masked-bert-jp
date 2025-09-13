# -*- coding: utf-8 -*-
"""Phase1→Phase2/3 の自動カリキュラム学習スクリプト

usage:
  python train_f123.py \
    --train data/train.jsonl \
    --val   data/val.jsonl \
    --model_name cl-tohoku/bert-base-japanese-v2 \
    --device cuda:0

入力 JSONL 形式:
{"id": "doc1", "sentences": ["文1", "文2", ...]}
"""
# ---- ユーザ提供コードをそのままほぼ転載 ----
import os, json, math, random, argparse
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

# ---- Utils ----

def set_seed(seed=42):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def clamp01(x): return max(0.0, min(1.0, x))

def sample_around(base: float, width: float = 0.1):
    lo = clamp01(base - width); hi = clamp01(base + width)
    lo = max(lo, 0.15); hi = min(hi, 0.95)
    return random.uniform(lo, hi)

# -------------------------
# Data utilities (Step-1)
# -------------------------

def mask_spans(input_ids: torch.Tensor, tokenizer: AutoTokenizer, mask_rate: float, span_mean: int = 8):
    """T5風 span corruption。masked_ids, labels, attn_mask を返す。"""
    B, T = input_ids.shape
    masked = input_ids.clone()
    labels = torch.full_like(input_ids, -100)
    for b in range(B):
        valid_pos = (input_ids[b] != tokenizer.pad_token_id).nonzero(as_tuple=False).squeeze(-1).tolist()
        if not valid_pos:
            continue
        n_to_mask = max(1, int(len(valid_pos) * mask_rate))
        covered = 0; tries = 0
        while covered < n_to_mask and tries < 10 * T:
            span = max(1, int(random.expovariate(1.0 / span_mean)))
            start = random.choice(valid_pos)
            end = min(start + span, T)
            # skip if overlap
            if (labels[b, start:end] != -100).any():
                tries += 1; continue
            masked[b, start:end] = tokenizer.mask_token_id
            labels[b, start:end] = input_ids[b, start:end]
            covered += (end - start)
            tries += 1
    attn_mask = (input_ids != tokenizer.pad_token_id).long()
    return masked, labels, attn_mask


class JsonlDocDataset(Dataset):
    """JSONL: {"id": str, "sentences": [str, ...]}"""
    def __init__(self, path: str):
        self.items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                self.items.append(json.loads(line))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


@dataclass
class Batch:
    input_ids: torch.Tensor        # (B, T)
    labels: torch.Tensor           # (B, T)
    attention_mask: torch.Tensor   # (B, T)
    ctx_sentences: List[List[str]] # per sample list of context sentences


def build_block(sentences: List[str], m: int):
    """連続 m 文をターゲットにし、それ以外を文脈とする。"""
    if not sentences:
        sentences = ["."]
    m = max(1, min(m, len(sentences)))
    start = random.randrange(len(sentences) - m + 1)
    tgt = sentences[start:start+m]
    others = sentences[:start] + sentences[start+m:]
    return tgt, others


def collate_fn(batch, tokenizer: AutoTokenizer, phase: int, m: int, mask_base: float, max_len_block: int):
    texts, ctxs = [], []
    sep_token = tokenizer.additional_special_tokens[0] if tokenizer.additional_special_tokens else "[SENT_SEP]"

    for ex in batch:
        sents = ex.get("sentences", []) or ["."]
        if phase == 1:
            tgt, others = build_block(sents, 1)
            ctxs.append([])
        else:
            tgt, others = build_block(sents, m)
            ctxs.append(others)
        block_text = f" {sep_token} ".join(tgt)
        texts.append(block_text)

    enc = tokenizer(texts, padding=True, truncation=True, max_length=max_len_block, return_tensors="pt")

    # マスク率をサンプリング
    mask_rate = sample_around(mask_base, 0.1)
    if mask_base >= 0.85 and random.random() < 0.05:
        mask_rate = 1.0

    masked_ids, labels, attention_mask = mask_spans(enc["input_ids"], tokenizer, mask_rate=mask_rate)

    return Batch(masked_ids, labels, attention_mask, ctxs)

# -------------------------
# Model blocks (Step-2)
# -------------------------


class TransformerSentenceEncoder(nn.Module):
    """シンプルな TransformerEncoder 一本で文(orブロック)を符号化"""
    def __init__(self, vocab_size: int, d_model: int = 768, n_heads: int = 12, n_layers: int = 6,
                 max_len: int = 512, pad_id: int = 0):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = nn.Embedding(max_len, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=4 * d_model,
                                               batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

    def forward(self, ids: torch.Tensor, attn_mask: torch.Tensor):
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device).unsqueeze(0).expand(B, T)
        x = self.tok(ids) + self.pos(pos)
        x = self.enc(x, src_key_padding_mask=(attn_mask == 0))
        return x  # (B,T,d)


class SlotResampler(nn.Module):
    """Perceiver-IO 風に学習可能な K スロットへ凝縮"""
    def __init__(self, d_model: int = 768, n_heads: int = 12, K: int = 8, n_layers: int = 2):
        super().__init__()
        self.K = K
        self.q = nn.Parameter(torch.randn(K, d_model) * 0.02)
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model, n_heads, dim_feedforward=4 * d_model,
                                       batch_first=True, norm_first=True)
            for _ in range(n_layers)
        ])

    def forward(self, sent_feats: torch.Tensor, sent_mask: torch.Tensor):
        """sent_feats: (B,S,d), sent_mask: (B,S)
        0=pad 1=valid
        """
        B, S, D = sent_feats.shape
        slots = self.q.unsqueeze(0).expand(B, self.K, D)  # (B,K,d)
        key_pad = (sent_mask == 0)
        for layer in self.layers:
            slots = layer(tgt=slots, memory=sent_feats, memory_key_padding_mask=key_pad)
        return slots  # (B,K,d)


class DocEncoder(nn.Module):
    """ターゲット以外の文を文脈スロット K 個へ圧縮"""

    def __init__(self, tokenizer: AutoTokenizer, d_model: int = 768, n_heads: int = 12,
                 n_layers: int = 4, K: int = 8, max_len_sent: int = 128):
        super().__init__()
        self.tok = tokenizer
        self.sent_enc = TransformerSentenceEncoder(tokenizer.vocab_size, d_model, n_heads, n_layers,
                                                   max_len=max_len_sent, pad_id=tokenizer.pad_token_id)
        self.resampler = SlotResampler(d_model, n_heads, K, n_layers=2)
        self.no_ctx = nn.Parameter(torch.randn(d_model) * 1e-3)
        self.K = K
        self.max_len_sent = max_len_sent

    @torch.no_grad()
    def _encode_sent_batch(self, list_of_lists, device):
        feats, masks = [], []
        for sents in list_of_lists:
            if len(sents) == 0:
                feats.append(torch.zeros(1, self.sent_enc.tok.embedding_dim, device=device))
                masks.append(torch.tensor([0], device=device))
                continue
            enc = self.tok(sents, padding=True, truncation=True, max_length=self.max_len_sent,
                            return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            h = self.sent_enc(enc["input_ids"], enc["attention_mask"])  # (S,T,d)
            mask = enc["attention_mask"].unsqueeze(-1)
            mean = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            feats.append(mean)  # (S,d)
            masks.append(torch.ones(mean.size(0), device=device))

        Smax = max(x.size(0) for x in feats)
        d = feats[0].size(-1)
        F = []; M = []
        for f, m in zip(feats, masks):
            padf = torch.zeros(Smax - f.size(0), d, device=device) if f.size(0) < Smax else torch.empty(0, d, device=device)
            padm = torch.zeros(Smax - m.size(0), device=device) if m.size(0) < Smax else torch.empty(0, device=device)
            F.append(torch.cat([f, padf], dim=0))
            M.append(torch.cat([m, padm], dim=0))
        return torch.stack(F, 0), torch.stack(M, 0)  # (B,Smax,d), (B,Smax)

    def forward(self, ctx_sentences, device, context_dropout: float = 0.2):
        sent_feats, sent_mask = self._encode_sent_batch(ctx_sentences, device)
        if sent_mask.sum() == 0:
            B = sent_feats.size(0)
            slots = self.no_ctx.view(1, 1, -1).expand(B, self.K, -1)
        else:
            slots = self.resampler(sent_feats, sent_mask)
        if context_dropout > 0:
            B, K, D = slots.shape
            drop = (torch.rand(B, K, device=slots.device) < context_dropout).unsqueeze(-1)
            slots = torch.where(drop, self.no_ctx.view(1, 1, -1).expand(B, K, D), slots)
        return slots  # (B,K,d)


class CrossAttnHead(nn.Module):
    def __init__(self, d_model: int, n_heads: int, vocab_size: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, h_tokens: torch.Tensor, doc_slots: torch.Tensor):
        # h_tokens: (B,T,d)  doc_slots: (B,K,d)
        attn_out, attn_w = self.attn(h_tokens, doc_slots, doc_slots)  # attn_w (B,T,K)
        h = self.ln(h_tokens + attn_out)
        logits = self.out(h)
        mean_slot_attn = attn_w.mean(dim=1).mean(dim=1)  # (B,)
        return logits, mean_slot_attn


class ContextConditionedMLM(nn.Module):
    """ターゲット文 MLM + 文脈スロット条件付け + InfoNCE"""

    def __init__(self, tokenizer: AutoTokenizer, d_model: int = 768, n_heads: int = 12,
                 enc_layers: int = 6, K: int = 8, max_len_block: int = 768):
        super().__init__()
        self.tokenizer = tokenizer
        self.sent_enc = TransformerSentenceEncoder(tokenizer.vocab_size, d_model, n_heads, enc_layers,
                                                   max_len=max_len_block, pad_id=tokenizer.pad_token_id)
        self.doc_enc = DocEncoder(tokenizer, d_model, n_heads, n_layers=4, K=K, max_len_sent=128)
        self.cross = CrossAttnHead(d_model, n_heads, tokenizer.vocab_size)
        self.proj_sent = nn.Linear(d_model, d_model)
        self.proj_slot = nn.Linear(d_model, d_model)

    def forward(self, batch: "Batch", context_dropout: float = 0.2):
        device = next(self.parameters()).device
        h = self.sent_enc(batch.input_ids.to(device), batch.attention_mask.to(device))  # (B,T,d)

        mask = batch.attention_mask.to(device).unsqueeze(-1)
        h_mean = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        slots = self.doc_enc(batch.ctx_sentences, device=device, context_dropout=context_dropout)  # (B,K,d)
        logits, mean_slot_attn = self.cross(h, slots)

        # InfoNCE (sentence vs slots)
        q = F.normalize(self.proj_sent(h_mean), dim=-1)
        k = F.normalize(self.proj_slot(slots.mean(dim=1)), dim=-1)
        logits_nce = q @ k.t() / 0.07
        nce = F.cross_entropy(logits_nce, torch.arange(q.size(0), device=device))
        return logits, mean_slot_attn, nce

# -------------------------
# Loss / metrics (Step-3)
# -------------------------


def mlm_loss(logits: torch.Tensor, labels: torch.Tensor):
    V = logits.size(-1)
    loss_ce = F.cross_entropy(logits.view(-1, V), labels.view(-1), ignore_index=-100)
    with torch.no_grad():
        preds = logits.argmax(-1)
        mask = (labels != -100)
        acc = (preds[mask] == labels[mask]).float().mean().item() if mask.any() else 0.0
    return loss_ce, acc


def use_reg(mean_slot_attn: torch.Tensor, alpha: float = 0.2, lam: float = 0.1):
    gap = (alpha - mean_slot_attn).clamp(min=0)
    return lam * gap.mean()


class EMA:
    def __init__(self, module: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {n: p.data.clone() for n, p in module.named_parameters() if p.requires_grad}
        self.module = module

    @torch.no_grad()
    def update(self):
        for n, p in self.module.named_parameters():
            if p.requires_grad:
                self.shadow[n] = self.decay * self.shadow[n] + (1 - self.decay) * p.data

    @torch.no_grad()
    def apply(self):
        for n, p in self.module.named_parameters():
            if p.requires_grad and n in self.shadow:
                p.data.copy_(self.shadow[n])


# -------------------------
# Trainer utilities (Step-4)
# -------------------------


def run_epoch(model: ContextConditionedMLM, loader, optimizer, phase: int,
              context_dropout: float, attn_alpha: float, attn_lambda: float, nce_lambda: float,
              validate: bool = False, context_off: bool = False):
    model.train(not validate)
    device = next(model.parameters()).device

    tot_loss_tok = tot_tok = 0.0
    tot_acc = tot_docattn = tot_nce = tot_use = 0.0

    for batch in tqdm(loader, leave=False, desc=("val" if validate else "train")):
        # 強制文脈オフの場合
        saved_ctx = None
        if context_off:
            saved_ctx = batch.ctx_sentences
            batch.ctx_sentences = [[] for _ in saved_ctx]

        logits, mean_slot_attn, loss_nce = model(batch, context_dropout=context_dropout if phase >= 2 else 0.0)
        loss_ce, acc = mlm_loss(logits, batch.labels.to(device))
        loss_use = use_reg(mean_slot_attn, alpha=attn_alpha, lam=attn_lambda) if phase >= 2 else logits.new_tensor(0.0)

        loss = loss_ce + (nce_lambda * loss_nce if phase >= 2 else 0.0) + loss_use

        if not validate:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        mcount = (batch.labels != -100).sum().item()
        tot_loss_tok += loss_ce.item() * max(1, mcount)
        tot_tok += max(1, mcount)
        tot_acc += acc * max(1, mcount)
        tot_docattn += mean_slot_attn.mean().item()
        if phase >= 2:
            tot_nce += loss_nce.item()
            tot_use += loss_use.item()

        if context_off:
            batch.ctx_sentences = saved_ctx

    ppl = math.exp(tot_loss_tok / max(1, tot_tok)) if tot_tok > 0 else float("inf")
    return {
        "ppl": ppl,
        "acc": tot_acc / max(1, tot_tok),
        "doc_attn": tot_docattn / max(1, len(loader)),
        "nce": tot_nce / max(1, len(loader)) if phase >= 2 else 0.0,
        "use": tot_use / max(1, len(loader)) if phase >= 2 else 0.0,
    }


# -------------------------
# Main orchestration (Step-5)
# -------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--model_name", default="cl-tohoku/bert-base-japanese-v2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--d_model", type=int, default=768)
    ap.add_argument("--n_heads", type=int, default=12)
    ap.add_argument("--enc_layers", type=int, default=6)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--max_len_block", type=int, default=768)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--epochs_p1", type=int, default=2)
    ap.add_argument("--val_acc_p1_to_p2", type=float, default=0.80)
    ap.add_argument("--context_dropout", type=float, default=0.2)
    ap.add_argument("--attn_alpha", type=float, default=0.2)
    ap.add_argument("--attn_lambda", type=float, default=0.1)
    ap.add_argument("--nce_lambda", type=float, default=0.5)
    ap.add_argument("--m_max", type=int, default=4)
    ap.add_argument("--mask_base_start", type=float, default=0.5)
    ap.add_argument("--mask_base_step", type=float, default=0.05)
    ap.add_argument("--plateau_eps", type=float, default=0.002)
    ap.add_argument("--plateau_patience", type=int, default=2)
    ap.add_argument("--save_dir", default="ckpt_f123")
    args = ap.parse_args()

    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # 追加特殊トークン
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": "[MASK]"})
    if "[SENT_SEP]" not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": ["[SENT_SEP]", "[EOS]"]})

    train_ds = JsonlDocDataset(args.train)
    val_ds = JsonlDocDataset(args.val)

    def make_loader(ds, phase, m, mask_base, shuffle):
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          collate_fn=lambda b: collate_fn(
                              b, tokenizer, phase=phase, m=m, mask_base=mask_base,
                              max_len_block=args.max_len_block))

    model = ContextConditionedMLM(tokenizer, d_model=args.d_model, n_heads=args.n_heads,
                                  enc_layers=args.enc_layers, K=args.K, max_len_block=args.max_len_block).to(args.device)

    # resize embeddings if vocab was extended
    model.sent_enc.tok = model.sent_enc.tok.to(args.device)
    if tokenizer.vocab_size != model.sent_enc.tok.num_embeddings:
        model.sent_enc.tok = nn.Embedding(tokenizer.vocab_size, args.d_model, padding_idx=tokenizer.pad_token_id).to(args.device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    os.makedirs(args.save_dir, exist_ok=True)

    # ---- Phase 1 ----
    print("== Phase 1: single-sentence MLM with [NO_CTX] ==")
    m = 1
    mask_base = 0.25
    train_loader = make_loader(train_ds, phase=1, m=m, mask_base=mask_base, shuffle=True)
    val_loader = make_loader(val_ds, phase=1, m=m, mask_base=mask_base, shuffle=False)

    best_val_acc = 0.0
    for ep in range(1, args.epochs_p1 + 1):
        tr = run_epoch(model, train_loader, opt, phase=1, context_dropout=0.0,
                       attn_alpha=args.attn_alpha, attn_lambda=args.attn_lambda,
                       nce_lambda=args.nce_lambda, validate=False)
        va = run_epoch(model, val_loader, opt, phase=1, context_dropout=0.0,
                       attn_alpha=args.attn_alpha, attn_lambda=args.attn_lambda,
                       nce_lambda=args.nce_lambda, validate=True)
        print(f"[P1 ep{ep}] train ppl={tr['ppl']:.3f} acc={tr['acc']:.3f} | val ppl={va['ppl']:.3f} acc={va['acc']:.3f}")
        best_val_acc = max(best_val_acc, va["acc"])
        torch.save(model.state_dict(), os.path.join(args.save_dir, f"p1_ep{ep}.pt"))
        if va["acc"] >= args.val_acc_p1_to_p2:
            print("→ Phase 2へ移行条件達成")
            break

    # ---- Phase 2/3 ----
    print("== Phase 2/3: doc-slot conditioned MLM ==")
    m = 1
    mask_base = args.mask_base_start
    plateau_cnt = 0
    best_val_acc = 0.0
    global_epoch = 0

    while m <= args.m_max:
        global_epoch += 1
        train_loader = make_loader(train_ds, phase=2, m=m, mask_base=mask_base, shuffle=True)
        val_loader = make_loader(val_ds, phase=2, m=m, mask_base=mask_base, shuffle=False)

        tr = run_epoch(model, train_loader, opt, phase=2, context_dropout=args.context_dropout,
                       attn_alpha=args.attn_alpha, attn_lambda=args.attn_lambda,
                       nce_lambda=args.nce_lambda, validate=False)
        va = run_epoch(model, val_loader, opt, phase=2, context_dropout=args.context_dropout,
                       attn_alpha=args.attn_alpha, attn_lambda=args.attn_lambda,
                       nce_lambda=args.nce_lambda, validate=True)
        va_off = run_epoch(model, val_loader, opt, phase=2, context_dropout=args.context_dropout,
                           attn_alpha=args.attn_alpha, attn_lambda=args.attn_lambda,
                           nce_lambda=args.nce_lambda, validate=True, context_off=True)
        delta = (va_off["ppl"] - va["ppl"]) / max(1e-9, va["ppl"]) * 100.0

        print(f"[m={m} mb={mask_base:.2f} ep={global_epoch}] train ppl={tr['ppl']:.2f} acc={tr['acc']:.3f} | "
              f"val ppl={va['ppl']:.2f} acc={va['acc']:.3f} ΔPPL_ctx_off={delta:.1f}%")

        torch.save(model.state_dict(), os.path.join(args.save_dir, f"m{m}_mb{mask_base:.2f}_ep{global_epoch}.pt"))

        # plateau 判定
        if va["acc"] > best_val_acc + args.plateau_eps:
            best_val_acc = va["acc"]
            plateau_cnt = 0
        else:
            plateau_cnt += 1

        if plateau_cnt >= args.plateau_patience:
            if m < args.m_max:
                print("↪ plateau → m を +1、mask_base をリセット")
                m += 1
                mask_base = args.mask_base_start
                plateau_cnt = 0
                best_val_acc = 0.0
                continue
            else:
                print("plateau & m==m_max → 終了")
                break
        else:
            mask_base = min(0.95, mask_base + args.mask_base_step)

    torch.save(model.state_dict(), os.path.join(args.save_dir, "final.pt"))
    print("Saved final model →", os.path.join(args.save_dir, "final.pt"))


if __name__ == "__main__":
    main()
