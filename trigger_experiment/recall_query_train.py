"""recall_query_train.py — train the Phase B indexer on each query variant and compare.

Indexer architecture is copied verbatim from phaseb_train.py (DSA-lightning-indexer,
~200k params, KL-distilled from teacher attention) so the only difference between the
three reported numbers is the query feature. Also reports raw cosine / raw QK top-2 of
each query against the contextual keys, for an untrained reference.

Run: python3 recall_query_train.py --data recall_query_labels_0.npz \
        --evaldata recall_query_labels_1.npz
"""
import argparse, glob, sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

VARIANTS = ("q_fullq", "q_spq", "q_spans")


def load(pattern):
    rows = []
    for f in sorted(glob.glob(pattern)):
        z = np.load(f)
        n = int(z["n_samples"])
        for j in range(n):
            rows.append({k: z[f"{j}_{k}"] for k in
                         ("labels", "k", "gold", "nB", *VARIANTS)})
    return rows


class Indexer(nn.Module):
    def __init__(self, L, Hq, Hkv, D, d=32):
        super().__init__()
        self.L, self.Hq, self.Hkv, self.D, self.d = L, Hq, Hkv, D, d
        self.group = Hq // Hkv
        self.Aq = nn.Parameter(torch.randn(L, Hq, D, d) * D ** -0.5)
        self.Bk = nn.Parameter(torch.randn(L, Hkv, 2 * D, d) * (2 * D) ** -0.5)
        self.w = nn.Parameter(torch.zeros(L, Hq))

    def forward(self, q, k):
        qp = torch.einsum("lhd,lhde->lhe", q, self.Aq)
        kp = torch.einsum("blkf,lkfe->blke", k, self.Bk)
        kp = kp.repeat_interleave(self.group, dim=2)
        sim = F.relu(torch.einsum("lhe,blhe->blh", qp, kp))
        return torch.einsum("blh,lh->b", sim,
                            torch.softmax(self.w.view(-1), 0).view(self.w.shape))


def raw_topk(rows, qkey, kind, K=2):
    """Untrained reference: cosine on mean-key, or raw QK (Phase A) top-k."""
    hit = 0
    for r in rows:
        q = torch.tensor(r[qkey], dtype=torch.float32)        # [L,Hq,D]
        k = torch.tensor(r["k"], dtype=torch.float32)         # [nB,L,Hkv,2D]
        nB = k.shape[0]
        group = q.shape[1] // k.shape[2]
        if kind == "qk":
            kk = k[..., :q.shape[-1]].repeat_interleave(group, dim=2)  # mean-key half
            sim = torch.einsum("lhd,blhd->blh", q, kk) / q.shape[-1] ** 0.5
            s = sim.mean((1, 2))                                       # [nB] mean heads/layers
        else:  # cosine on head-averaged per-layer vectors (GQA-safe)
            qf = q.mean(1).reshape(-1)                            # [L*D]
            kf = k[..., :q.shape[-1]].mean(2).reshape(nB, -1)     # [nB, L*D]
            s = F.cosine_similarity(kf, qf[None], dim=1)
        hit += int(r["gold"]) in s.topk(min(K, nB)).indices.tolist()
    return hit / len(rows)


def train_variant(tr, val, ho, qkey, epochs, lr, dev):
    L, Hq, D = tr[0][qkey].shape
    Hkv = tr[0]["k"].shape[2]
    model = Indexer(L, Hq, Hkv, D).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def tensors(r):
        q = torch.tensor(r[qkey], dtype=torch.float32, device=dev)
        k = torch.tensor(r["k"], dtype=torch.float32, device=dev)
        lab = torch.tensor(r["labels"].mean(0), dtype=torch.float32, device=dev)
        lab = lab / max(lab.sum().item(), 1e-8)
        return q, k, lab

    def top2(split):
        hit = 0
        with torch.no_grad():
            for r in split:
                q, k, _ = tensors(r)
                s = model(q, k)
                hit += int(r["gold"]) in s.topk(min(2, len(s))).indices.tolist()
        return hit / len(split)

    rng = np.random.RandomState(0)
    best, best_state = -1, None
    for ep in range(epochs):
        rng.shuffle(tr)
        for r in tr:
            q, k, lab = tensors(r)
            s = model(q, k)
            loss = F.kl_div(F.log_softmax(s, 0), lab, reduction="sum")
            opt.zero_grad(); loss.backward(); opt.step()
        v = top2(val)
        if v > best:
            best = v
            best_state = {kk: vv.detach().cpu().clone()
                          for kk, vv in model.state_dict().items()}
    model.load_state_dict(best_state)
    return best, (top2(ho) if ho else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--evaldata", default="")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--val", type=float, default=0.2)
    args = ap.parse_args()
    rows = load(args.data)
    ho = load(args.evaldata) if args.evaldata else []
    print(f"RECALLQ_TRAIN samples={len(rows)} heldout={len(ho)}", flush=True)
    if len(rows) < 40:
        print("not enough samples"); return
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(rows))
    n_val = int(len(rows) * args.val)
    val = [rows[i] for i in idx[:n_val]]
    tr = [rows[i] for i in idx[n_val:]]
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    print("--- untrained reference (heldout top-2) ---", flush=True)
    ref = ho or val
    for v in VARIANTS:
        print(f"  {v:8s} raw_cos={raw_topk(ref, v, 'cos'):.3f} "
              f"raw_qk={raw_topk(ref, v, 'qk'):.3f}", flush=True)

    print("--- learned indexer ---", flush=True)
    results = {}
    for v in VARIANTS:
        bv, ht = train_variant([dict(r) for r in tr], val, ho, v, args.epochs,
                               args.lr, dev)
        results[v] = (bv, ht)
        print(f"  {v:8s} val_top2={bv:.3f} heldout_top2="
              f"{ht if ht is None else round(ht, 3)}", flush=True)
    print("RECALLQ_TRAIN_DONE " +
          " ".join(f"{v}={results[v][1] if results[v][1] is not None else results[v][0]:.3f}"
                   for v in VARIANTS), flush=True)


if __name__ == "__main__":
    main()
