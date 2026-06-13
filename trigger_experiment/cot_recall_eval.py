"""cot_recall_eval.py — evaluate the two questions of COTRECALL on the same battery.

Q1 (retrieval): does the FIRE-position query retrieve the gold fact block better than
   the question-position query? Train the Phase B indexer on each (same keys/labels),
   report heldout top-1/top-2. Also raw qk reference.

Q2 (mid-CoT gate): can a per-position gate fire here? (a) teacher attention to evicted
   spikes at the fire position vs a control CoT position (mean fire_mass vs ctrl_mass);
   (b) a linear probe on SP-context features separates fire from control (AUC).

Run: python3 cot_recall_eval.py --data cot_recall_labels_0.npz \
        --evaldata cot_recall_labels_1.npz
"""
import argparse, glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

QVARIANTS = ("q_question", "q_fire_generic", "q_fire")


def load(pattern):
    rows = []
    for f in sorted(glob.glob(pattern)):
        z = np.load(f)
        n = int(z["n_samples"])
        for j in range(n):
            rows.append({k: z[f"{j}_{k}"] for k in
                         ("labels", "k", "gold", "nB", "q_question", "q_fire",
                          "q_fire_generic", "q_ctrl", "fire_mass", "ctrl_mass")})
    return rows


class Indexer(nn.Module):
    def __init__(self, L, Hq, Hkv, D, d=32):
        super().__init__()
        self.group = Hq // Hkv
        self.Aq = nn.Parameter(torch.randn(L, Hq, D, d) * D ** -0.5)
        self.Bk = nn.Parameter(torch.randn(L, Hkv, 2 * D, d) * (2 * D) ** -0.5)
        self.w = nn.Parameter(torch.zeros(L, Hq))

    def forward(self, q, k):
        qp = torch.einsum("lhd,lhde->lhe", q, self.Aq)
        kp = torch.einsum("blkf,lkfe->blke", k, self.Bk).repeat_interleave(self.group, 2)
        sim = F.relu(torch.einsum("lhe,blhe->blh", qp, kp))
        return torch.einsum("blh,lh->b", sim,
                            torch.softmax(self.w.view(-1), 0).view(self.w.shape))


def raw_qk_topk(rows, qkey, K):
    hit = 0
    for r in rows:
        q = torch.tensor(r[qkey], dtype=torch.float32)
        k = torch.tensor(r["k"], dtype=torch.float32)
        group = q.shape[1] // k.shape[2]
        kk = k[..., :q.shape[-1]].repeat_interleave(group, dim=2)
        s = torch.einsum("lhd,blhd->blh", q, kk).mean((1, 2))
        hit += int(r["gold"]) in s.topk(min(K, k.shape[0])).indices.tolist()
    return hit / len(rows)


def train_indexer(tr, val, ho, qkey, seeds=3, epochs=40, lr=3e-3):
    L, Hq, D = tr[0][qkey].shape
    Hkv = tr[0]["k"].shape[2]
    t1s, t2s = [], []
    for sd in range(seeds):
        torch.manual_seed(sd)
        model = Indexer(L, Hq, Hkv, D)
        opt = torch.optim.Adam(model.parameters(), lr=lr)

        def tens(r):
            q = torch.tensor(r[qkey], dtype=torch.float32)
            k = torch.tensor(r["k"], dtype=torch.float32)
            lab = torch.tensor(r["labels"].mean(0), dtype=torch.float32)
            return q, k, lab / max(lab.sum().item(), 1e-8)

        def topk(split, K):
            h = 0
            with torch.no_grad():
                for r in split:
                    q, k, _ = tens(r)
                    h += int(r["gold"]) in model(q, k).topk(min(K, len(k))).indices.tolist()
            return h / len(split)

        rng = np.random.RandomState(sd)
        best, best_state = -1, None
        for _ in range(epochs):
            rng.shuffle(tr)
            for r in tr:
                q, k, lab = tens(r)
                loss = F.kl_div(F.log_softmax(model(q, k), 0), lab, reduction="sum")
                opt.zero_grad(); loss.backward(); opt.step()
            v = topk(val, 2)
            if v > best:
                best = v
                best_state = {kk: vv.clone() for kk, vv in model.state_dict().items()}
        model.load_state_dict(best_state)
        ref = ho or val
        t1s.append(topk(ref, 1)); t2s.append(topk(ref, 2))
    return np.mean(t1s), np.std(t1s), np.mean(t2s), np.std(t2s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--evaldata", default="")
    args = ap.parse_args()
    rows = load(args.data)
    ho = load(args.evaldata) if args.evaldata else []
    print(f"COTRECALL_EVAL train={len(rows)} heldout={len(ho)}", flush=True)
    if len(rows) < 40:
        print("not enough samples"); return
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(rows))
    nv = int(len(rows) * 0.2)
    val = [rows[i] for i in idx[:nv]]
    tr = [rows[i] for i in idx[nv:]]
    ref = ho or val

    print("--- Q2 mid-CoT gate ---", flush=True)
    fm = np.array([float(r["fire_mass"]) for r in rows])
    cm = np.array([float(r["ctrl_mass"]) for r in rows])
    print(f"  teacher attn->evicted: fire {fm.mean():.3f}  ctrl {cm.mean():.3f}  "
          f"(fire>ctrl in {(fm > cm).mean()*100:.0f}% of samples)")
    Xf = np.array([r["q_fire"].reshape(-1) for r in tr], np.float32)
    Xc = np.array([r["q_ctrl"].reshape(-1) for r in tr], np.float32)
    X = np.vstack([Xf, Xc]); y = np.r_[np.ones(len(Xf)), np.zeros(len(Xc))]
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=0.03, max_iter=3000).fit(sc.transform(X), y)
    Xfe = np.array([r["q_fire"].reshape(-1) for r in ref], np.float32)
    Xce = np.array([r["q_ctrl"].reshape(-1) for r in ref], np.float32)
    Xe = np.vstack([Xfe, Xce]); ye = np.r_[np.ones(len(Xfe)), np.zeros(len(Xce))]
    auc = roc_auc_score(ye, clf.decision_function(sc.transform(Xe)))
    print(f"  SP-context probe fire-vs-control AUC (heldout): {auc:.3f}")

    print("--- Q1 retrieval (heldout top-1 / top-2, 3-seed) ---", flush=True)
    for v in QVARIANTS:
        print(f"  {v:11s} raw_qk top2={raw_qk_topk(ref, v, 2):.3f}", flush=True)
    res = {}
    for v in QVARIANTS:
        t1, s1, t2, s2 = train_indexer([dict(r) for r in tr], val, ho, v)
        res[v] = (t1, t2)
        print(f"  {v:11s} indexer top1={t1:.3f}±{s1:.3f}  top2={t2:.3f}±{s2:.3f}",
              flush=True)
    print("COTRECALL_EVAL_DONE " +
          " ".join(f"{v}_top2={res[v][1]:.3f}" for v in QVARIANTS), flush=True)


if __name__ == "__main__":
    main()
