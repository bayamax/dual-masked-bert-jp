"""phaseb_train.py — train the Phase B recall indexer on teacher labels (PHASEB_V1).

Indexer (DSA-lightning-indexer-shaped, ~200k params):
    per layer l: q'_h = A_l q_h,  k'_h = B_l k_feat_h   (D->d projections; k_feat=2D)
    score(block) = sum_l sum_h w_{l,h} * ReLU(q'_{l,h} . k'_{l,block,h})
    loss = KL( teacher_label_agg || softmax(scores) )
Teacher label: per-layer attention-mass labels averaged over layers (already
sink-excluded, window-excluded, L1-normalized).

Metrics: gold-in-top2 rate on a held-out split (compare: BGE 7/10, raw-QK 1/10 on the
needle bench; ceiling = full KV).

Run: python3 phaseb_train.py --data 'phaseb_labels_*.npz' --epochs 40
Exports: phaseb_indexer.npz (numpy-only, LinearHead-style, for the runtime).
"""
import argparse, glob, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def load(pattern):
    rows = []
    for f in sorted(glob.glob(pattern)):
        z = np.load(f)
        n = int(z["n_samples"])
        for j in range(n):
            rows.append({k: z[f"{j}_{k}"] for k in ("labels", "q", "k", "gold", "nB")})
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
        # q: [L,Hq,D]  k: [nB,L,Hkv,2D]  -> scores [nB]
        qp = torch.einsum("lhd,lhde->lhe", q, self.Aq)             # [L,Hq,d]
        kp = torch.einsum("blkf,lkfe->blke", k, self.Bk)           # [nB,L,Hkv,d]
        kp = kp.repeat_interleave(self.group, dim=2)               # [nB,L,Hq,d]
        sim = F.relu(torch.einsum("lhe,blhe->blh", qp, kp))
        return torch.einsum("blh,lh->b", sim, torch.softmax(self.w.view(-1), 0).view(self.w.shape))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="phaseb_labels_*.npz")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--val", type=float, default=0.2)
    ap.add_argument("--out", default="phaseb_indexer.npz")
    ap.add_argument("--evaldata", default="")
    args = ap.parse_args()
    rows = load(args.data)
    print(f"PHASEB_TRAIN_V1 samples={len(rows)}", flush=True)
    if len(rows) < 40:
        print("not enough samples"); return
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(rows))
    n_val = int(len(rows) * args.val)
    val, tr = [rows[i] for i in idx[:n_val]], [rows[i] for i in idx[n_val:]]
    L, Hq, D = rows[0]["q"].shape
    Hkv = rows[0]["k"].shape[2]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = Indexer(L, Hq, Hkv, D).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    def tensors(r):
        q = torch.tensor(r["q"], dtype=torch.float32, device=dev)
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

    best, best_state = -1, None
    for ep in range(args.epochs):
        rng.shuffle(tr)
        tot = 0.0
        for r in tr:
            q, k, lab = tensors(r)
            s = model(q, k)
            loss = F.kl_div(F.log_softmax(s, 0), lab, reduction="sum")
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        v = top2(val)
        if v > best:
            best, best_state = v, {k2: v2.detach().cpu().clone()
                                   for k2, v2 in model.state_dict().items()}
        print(f"ep{ep:02d} loss={tot/len(tr):.4f} val_top2={v:.3f} (best {best:.3f})",
              flush=True)
    model.load_state_dict(best_state)
    if args.evaldata:
        ho = load(args.evaldata)
        if ho:
            print(f"HELDOUT top2={top2(ho):.3f} n={len(ho)}", flush=True)
    np.savez(args.out, **{k2: v2.numpy() for k2, v2 in best_state.items()},
             meta=np.array([L, Hq, Hkv, D, model.d]))
    print(f"PHASEB_TRAIN_DONE best_val_top2={best:.3f} train_top2={top2(tr):.3f}",
          flush=True)


if __name__ == "__main__":
    main()
