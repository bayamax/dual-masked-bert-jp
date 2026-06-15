"""train_components.py — train & SAVE the two correct COTSCAN components from the natural
mid-CoT labels, so the task eval can wire them into generation:

  * gate probe  : logistic on SP-context pre-RoPE q, fire (value-emission pos) vs control.
  * Phase B indexer : learned projection scoring evicted blocks from the fire-pos query.

Inputs: cot_recall_natural_0.npz (train) + _1.npz (heldout) produced by cot_recall_natural.py.
Output: task_components.npz (gate g_* + indexer ix_* + dims).
"""
import os, sys, json
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cot_recall_eval import load, Indexer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


def train_indexer_save(tr, val, qkey="q_fire", seeds=3, epochs=40, lr=3e-3):
    L, Hq, D = tr[0][qkey].shape
    Hkv = tr[0]["k"].shape[2]

    def tens(r):
        q = torch.tensor(r[qkey], dtype=torch.float32)
        k = torch.tensor(r["k"], dtype=torch.float32)
        lab = torch.tensor(r["labels"].mean(0), dtype=torch.float32)
        return q, k, lab / max(lab.sum().item(), 1e-8)

    def top2(model, split):
        h = 0
        with torch.no_grad():
            for r in split:
                q, k, _ = tens(r)
                h += int(r["gold"]) in model(q, k).topk(min(2, len(k))).indices.tolist()
        return h / len(split)

    best_overall, best_state = -1, None
    for sd in range(seeds):
        torch.manual_seed(sd)
        model = Indexer(L, Hq, Hkv, D)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        rng = np.random.RandomState(sd)
        best, state = -1, None
        for _ in range(epochs):
            rng.shuffle(tr)
            for r in tr:
                q, k, lab = tens(r)
                loss = F.kl_div(F.log_softmax(model(q, k), 0), lab, reduction="sum")
                opt.zero_grad(); loss.backward(); opt.step()
            v = top2(model, val)
            if v > best:
                best = v; state = {kk: vv.clone() for kk, vv in model.state_dict().items()}
        if best > best_overall:
            best_overall = best; best_state = state
    return best_state, best_overall, (L, Hq, Hkv, D)


def main():
    tr = load("cot_recall_natural_0.npz")
    ho = load("cot_recall_natural_1.npz")
    print(f"train={len(tr)} heldout={len(ho)}", flush=True)

    # ---- gate probe: fire (q_fire, label 1) vs control (q_ctrl, label 0) ----
    Xf = np.array([r["q_fire"].reshape(-1) for r in tr], np.float32)
    Xc = np.array([r["q_ctrl"].reshape(-1) for r in tr], np.float32)
    X = np.vstack([Xf, Xc]); y = np.r_[np.ones(len(Xf)), np.zeros(len(Xc))]
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=0.03, max_iter=3000).fit(sc.transform(X), y)
    Hf = np.array([r["q_fire"].reshape(-1) for r in ho], np.float32)
    Hc = np.array([r["q_ctrl"].reshape(-1) for r in ho], np.float32)
    He = np.vstack([Hf, Hc]); ye = np.r_[np.ones(len(Hf)), np.zeros(len(Hc))]
    auc = float(roc_auc_score(ye, clf.decision_function(sc.transform(He))))
    posS = clf.decision_function(sc.transform(Xf))
    # operating point: median of positives (TPR~0.5) — conservative so the per-position
    # gate fires near the genuine recall moment, not constantly.
    thresh = float(np.median(posS))
    print(f"[gate] heldout fire-vs-ctrl AUC={auc:.3f} thresh={thresh:.3f}", flush=True)

    # ---- Phase B indexer on the fire-position query ----
    rng = np.random.RandomState(0); idx = rng.permutation(len(tr)); nv = int(len(tr) * 0.2)
    val = [tr[i] for i in idx[:nv]]; tr2 = [dict(tr[i]) for i in idx[nv:]]
    state, best_val, (L, Hq, Hkv, D) = train_indexer_save(tr2, val)
    # heldout top-2 with the saved indexer
    m = Indexer(L, Hq, Hkv, D); m.load_state_dict(state)
    def top2(split):
        h = 0
        with torch.no_grad():
            for r in split:
                q = torch.tensor(r["q_fire"], dtype=torch.float32)
                k = torch.tensor(r["k"], dtype=torch.float32)
                h += int(r["gold"]) in m(q, k).topk(min(2, len(k))).indices.tolist()
        return h / len(split)
    ho_top2 = top2(ho)
    print(f"[indexer] val_top2={best_val:.3f} heldout_top2={ho_top2:.3f}", flush=True)

    np.savez("task_components.npz",
             g_coef=clf.coef_.ravel().astype(np.float32),
             g_intercept=np.float32(clf.intercept_[0]),
             g_mean=sc.mean_.astype(np.float32), g_scale=sc.scale_.astype(np.float32),
             g_thresh=np.float32(thresh), g_auc=np.float32(auc),
             ix_Aq=state["Aq"].numpy(), ix_Bk=state["Bk"].numpy(), ix_w=state["w"].numpy(),
             L=np.int64(L), Hq=np.int64(Hq), Hkv=np.int64(Hkv), D=np.int64(D),
             ho_top2=np.float32(ho_top2))
    json.dump({"gate_auc": auc, "gate_thresh": thresh, "indexer_heldout_top2": ho_top2,
               "indexer_val_top2": best_val, "n_train": len(tr), "n_heldout": len(ho)},
              open("components_summary.json", "w"), indent=2)
    print("COMPONENTS_SAVED", flush=True)


if __name__ == "__main__":
    main()
