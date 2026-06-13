"""explore.py — trial-and-error improvements on the validated recall stack.

Reuses the Dolphin labelling pipeline + the trained recall_kit_v4 weights to push on the
weakest link (the GATE) and to quantify the user's backward-scan runtime idea.

(A) GATE operating point: PR curve on heldout -> AUC, best-F1 threshold, recall@P=0.9,
    precision@R=0.9, and the shipped-threshold point. (the gate's recall was the weak link)
(B) GATE model search: logistic vs MLP vs bigger-MLP heldout AUC (can we beat 0.958?).
(C) Backward streaming scan (BGE learned head, state-free): scan evicted blocks recent->old,
    stop at first score over a calibrated threshold. Report reach-rate, avg blocks scanned,
    and the gold block's recency-rank distribution ("how far back is the answer").
(D) BGE-small vs base for the elegant path (runtime weight).
"""
import os, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dolphin_scan as DS
from recall_kit import RecallGate, BGERetriever
from recall_kit.retriever import BGEHead

OUT = os.environ.get("OUT_DIR", "results_explore")
DEV = DS.DEV
LOG = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.append(s)
    os.makedirs(OUT, exist_ok=True); open(os.path.join(OUT, "explore.log"), "w").write("\n".join(LOG))


def gate_operating_point(score_te, yte, shipped_thresh):
    from sklearn.metrics import precision_recall_curve, roc_auc_score
    auc = float(roc_auc_score(yte, score_te))
    P, R, T = precision_recall_curve(yte, score_te)
    f1 = 2 * P * R / np.clip(P + R, 1e-9, None)
    bi = int(np.nanargmax(f1))
    def r_at_p(p0):
        ok = np.where(P[:-1] >= p0)[0]
        return float(R[ok].max()) if len(ok) else 0.0
    def p_at_r(r0):
        ok = np.where(R[:-1] >= r0)[0]
        return float(P[ok].max()) if len(ok) else 0.0
    sh_pred = score_te > shipped_thresh
    sh_p = float((sh_pred & (yte == 1)).sum() / max(sh_pred.sum(), 1))
    sh_r = float((sh_pred & (yte == 1)).sum() / max((yte == 1).sum(), 1))
    return {"auc": round(auc, 4),
            "best_f1": {"f1": round(float(f1[bi]), 3), "precision": round(float(P[bi]), 3),
                        "recall": round(float(R[bi]), 3),
                        "thresh": round(float(T[bi]) if bi < len(T) else 0.0, 3)},
            "recall_at_p0.9": round(r_at_p(0.9), 3),
            "precision_at_r0.9": round(p_at_r(0.9), 3),
            "shipped_threshold": {"thresh": round(shipped_thresh, 3),
                                  "precision": round(sh_p, 3), "recall": round(sh_r, 3)}}


class MLP(nn.Module):
    def __init__(self, d_in, h=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, h), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(h, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp_gate(Xtr, ytr, Xte, yte, h=256, epochs=8):
    from sklearn.metrics import roc_auc_score
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr_ = torch.tensor((Xtr - mu) / sd, dtype=torch.float32, device=DEV)
    Xte_ = torch.tensor((Xte - mu) / sd, dtype=torch.float32, device=DEV)
    ytr_ = torch.tensor(ytr, dtype=torch.float32, device=DEV)
    pos_w = torch.tensor([(ytr == 0).sum() / max((ytr == 1).sum(), 1)], device=DEV)
    m = MLP(Xtr.shape[1], h).to(DEV); opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    n = len(ytr_); bs = 4096
    for ep in range(epochs):
        idx = torch.randperm(n, device=DEV)
        for i in range(0, n, bs):
            b = idx[i:i + bs]
            loss = F.binary_cross_entropy_with_logits(m(Xtr_[b]), ytr_[b], pos_weight=pos_w)
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        s = m(Xte_).cpu().numpy()
    return float(roc_auc_score(yte, s))


def backward_scan(IXte, bge_head, tau_list, recent_first=True):
    """State-free recent->old scan: stop at first block with head-score>tau. Returns metrics."""
    recency_ranks = []        # nB-1-gold (0 = gold is the most-recent evicted block)
    out = {}
    # precompute per-row head scores over all blocks
    rows = []
    with torch.no_grad():
        for r in IXte:
            q = torch.tensor(r["q"].reshape(-1), dtype=torch.float32, device=DEV)
            k = torch.tensor(r["bge_k"], dtype=torch.float32, device=DEV)
            s = bge_head(q, k).cpu().numpy()           # [nB]
            rows.append((s, int(r["gold"]), int(r["nB"])))
            recency_ranks.append(int(r["nB"]) - 1 - int(r["gold"]))
    for tau in tau_list:
        reach = 0; scanned = []
        for s, gold, nB in rows:
            order = range(nB - 1, -1, -1) if recent_first else range(nB)
            stop = None; cnt = 0
            for b in order:
                cnt += 1
                if s[b] > tau:
                    stop = b; break
            scanned.append(cnt)
            if stop == gold:
                reach += 1
            elif stop is None and int(np.argmax(s)) == gold:
                pass
        out[f"tau={tau:.2f}"] = {"reach_rate": round(reach / len(rows), 3),
                                 "avg_blocks_scanned": round(float(np.mean(scanned)), 2)}
    rr = np.array(recency_ranks)
    out["gold_recency_rank"] = {"mean": round(float(rr.mean()), 2),
                                "median": int(np.median(rr)),
                                "frac_in_recent_3": round(float((rr < 3).mean()), 3),
                                "frac_in_recent_8": round(float((rr < 8).mean()), 3)}
    return out


def main():
    t0 = time.time(); os.makedirs(OUT, exist_ok=True)
    tok, llm, pooler, embT, bge = DS.build_model()
    cfg = llm.config; dims = (cfg.num_attention_heads, cfg.num_key_value_heads,
                              cfg.hidden_size // cfg.num_attention_heads)
    N_TR = int(os.environ.get("EXP_N_TRAIN", "150")); N_HE = int(os.environ.get("EXP_N_HELD", "80"))
    pool = DS.pick_items(N_TR + N_HE)
    tr_pool, te_pool = pool[:N_TR], pool[N_TR:]
    log(f"[data] train {len(tr_pool)} held {len(te_pool)}")
    Xtr, ytr, _, IXtr = DS.collect(tr_pool, llm, tok, pooler, embT, dims, "TR", t0, bge=bge)
    Xte, yte, _, IXte = DS.collect(te_pool, llm, tok, pooler, embT, dims, "HE", t0, bge=bge)
    log(f"[data] gate pts tr {len(ytr)}({int(ytr.sum())}+) te {len(yte)}({int(yte.sum())}+) "
        f"idx te {len(IXte)}")
    res = {"data": {"gate_pts_held": int(len(yte)), "pos_held": int(yte.sum()),
                    "idx_held": len(IXte)}}

    gate = RecallGate.load("art/gate.npz")
    score_te = np.array([gate.score(x) for x in Xte])
    res["A_gate_operating_point"] = gate_operating_point(score_te, yte, gate.thresh)
    log(f"[A] gate AUC={res['A_gate_operating_point']['auc']} "
        f"best-F1={res['A_gate_operating_point']['best_f1']} "
        f"R@P0.9={res['A_gate_operating_point']['recall_at_p0.9']} "
        f"shipped={res['A_gate_operating_point']['shipped_threshold']}")

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(C=0.05, max_iter=2000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
    auc_lr = float(roc_auc_score(yte, lr.decision_function(sc.transform(Xte))))
    auc_mlp = train_mlp_gate(Xtr, ytr, Xte, yte, h=256)
    auc_mlp2 = train_mlp_gate(Xtr, ytr, Xte, yte, h=512, epochs=12)
    res["B_gate_model_search"] = {"logistic_auc": round(auc_lr, 4), "mlp256_auc": round(auc_mlp, 4),
                                  "mlp512_auc": round(auc_mlp2, 4)}
    log(f"[B] gate AUC  logistic={auc_lr:.4f} mlp256={auc_mlp:.4f} mlp512={auc_mlp2:.4f}")

    bge_ret = BGERetriever.load("art/bge_head.npz", bge, device=DEV)
    # calibrate tau from train gold-block head-scores
    gs = []
    with torch.no_grad():
        for r in IXtr[:1500]:
            q = torch.tensor(r["q"].reshape(-1), dtype=torch.float32, device=DEV)
            k = torch.tensor(r["bge_k"], dtype=torch.float32, device=DEV)
            gs.append(float(bge_ret.head(q, k)[int(r["gold"])]))
    gs = np.array(gs)
    taus = [float(np.quantile(gs, q)) for q in (0.1, 0.25, 0.5)]
    res["C_backward_scan"] = backward_scan(IXte, bge_ret.head, taus)
    log(f"[C] backward-scan {json.dumps(res['C_backward_scan'])}")

    # (D) BGE-small for the elegant path: re-embed blocks with bge-small, score with a
    # quickly-retrained head, heldout top-2.
    try:
        from sentence_transformers import SentenceTransformer
        small = SentenceTransformer("BAAI/bge-small-en-v1.5", device=DEV)
        def reembed(rows):
            for r in rows:
                # we stored bge_k from base; for small we need block texts -> not stored.
                pass
        res["D_bge_small"] = {"note": "skipped (block texts not in IX rows); base bge_head is canonical"}
    except Exception as e:
        res["D_bge_small"] = {"error": str(e)[:80]}

    res["final"] = True; res["elapsed_s"] = round(time.time() - t0, 1)
    json.dump(res, open(os.path.join(OUT, "explore.json"), "w"), indent=2)
    log("\n=== EXPLORE SUMMARY ===\n" + json.dumps(res, indent=1))
    log(f"DONE t={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
