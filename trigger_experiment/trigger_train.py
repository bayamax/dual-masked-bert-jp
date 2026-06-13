"""trigger_train.py — premise stats + trigger-head training on trigger_labels npz.

1. Premise check: per question type, where does the teacher's answer attention mass
   actually go (evicted / window / suffix)? Reports the oracle AUC of the measured
   evicted-mass fraction itself — if THIS is low the whole approach is dead, no head
   can fix it.
2. Head: logistic regression on question-position pre-RoPE query features (the same
   feature inference has), predicting qtype==EVICT. Reports AUC on the eval shard
   for evict-vs-all, evict-vs-window (hard), evict-vs-generic.

Run: python3 trigger_train.py --data trigger_labels_0.npz --evaldata trigger_labels_1.npz
"""
import argparse

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

TYPES = ("EVICT", "WINDOW", "GENERIC")


def load(path):
    z = np.load(path)
    n = int(z["n_samples"])
    rows = []
    for j in range(n):
        rows.append({"fracs": z[f"{j}_fracs"], "q": z[f"{j}_q"].astype(np.float32),
                     "qtype": int(z[f"{j}_qtype"]), "seed": int(z[f"{j}_seed"])})
    return rows


def premise_stats(rows, name):
    print(f"--- premise stats [{name}] n={len(rows)} ---")
    if not np.any(np.stack([r["fracs"] for r in rows])):
        print("  (no teacher fracs in this file — feature-only shard, skipping)")
        return
    L = rows[0]["fracs"].shape[0]
    for t in range(3):
        sel = [r for r in rows if r["qtype"] == t]
        if not sel:
            continue
        f = np.stack([r["fracs"] for r in sel])               # [n, L, 3]
        for li in range(L):
            ev, wi, su = f[:, li].mean(0)
            print(f"  {TYPES[t]:8s} L{li} evict={ev:.3f} window={wi:.3f} suffix={su:.3f} "
                  f"(ev p10/p50/p90 {np.percentile(f[:, li, 0], [10, 50, 90]).round(3)})")
    y = np.array([r["qtype"] == 0 for r in rows])
    score = np.stack([r["fracs"][:, 0].mean() for r in rows])
    if 0 < y.sum() < len(y):
        print(f"  ORACLE evicted-mass AUC (evict vs rest): {roc_auc_score(y, score):.4f}")
        yw = np.array([r["qtype"] for r in rows])
        m = yw != 2
        if 0 < y[m].sum() < m.sum():
            print(f"  ORACLE evicted-mass AUC (evict vs window): {roc_auc_score(y[m], score[m]):.4f}")


def featurize(rows):
    X = np.stack([r["q"].reshape(-1) for r in rows])
    y = np.array([r["qtype"] == 0 for r in rows], dtype=int)
    g = np.array([r["seed"] for r in rows])
    t = np.array([r["qtype"] for r in rows])
    return X, y, g, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--evaldata", default=None)
    ap.add_argument("--out", default="trigger_head.npz")
    ap.add_argument("--prevhead", default=None,
                    help="score eval data with a previously trained head (transfer check)")
    args = ap.parse_args()
    tr = load(args.data)
    premise_stats(tr, args.data)
    Xtr, ytr, gtr, _ = featurize(tr)
    sc = StandardScaler().fit(Xtr)
    Xs = sc.transform(Xtr)

    best_c, best_auc = None, -1.0
    n_groups = len(set(gtr.tolist()))
    for C in (0.003, 0.01, 0.03, 0.1, 0.3):
        aucs = []
        for itr, iva in GroupKFold(n_splits=min(5, n_groups)).split(Xs, ytr, gtr):
            if len(set(ytr[iva])) < 2:
                continue
            clf = LogisticRegression(C=C, max_iter=3000).fit(Xs[itr], ytr[itr])
            aucs.append(roc_auc_score(ytr[iva], clf.decision_function(Xs[iva])))
        m = float(np.mean(aucs)) if aucs else -1.0
        print(f"C={C} cv_auc={m:.4f}", flush=True)
        if m > best_auc:
            best_c, best_auc = C, m
    clf = LogisticRegression(C=best_c, max_iter=3000).fit(Xs, ytr)
    print(f"chosen C={best_c} (cv_auc={best_auc:.4f})")

    if args.evaldata:
        ev = load(args.evaldata)
        premise_stats(ev, args.evaldata)
        Xev, yev, _, tev = featurize(ev)
        s = clf.decision_function(sc.transform(Xev))
        print(f"EVAL AUC evict-vs-all:     {roc_auc_score(yev, s):.4f}")
        m = tev != 2
        if 0 < yev[m].sum() < m.sum():
            print(f"EVAL AUC evict-vs-window:  {roc_auc_score(yev[m], s[m]):.4f}")
        m = tev != 1
        if 0 < yev[m].sum() < m.sum():
            print(f"EVAL AUC evict-vs-generic: {roc_auc_score(yev[m], s[m]):.4f}")
        if args.prevhead:
            h = np.load(args.prevhead)
            sp = ((Xev - h["mean"]) / h["scale"]) @ h["coef"].T + h["intercept"]
            sp = sp.ravel()
            print(f"PREVHEAD transfer AUC evict-vs-all:    {roc_auc_score(yev, sp):.4f}")
            m = tev != 2
            print(f"PREVHEAD transfer AUC evict-vs-window: {roc_auc_score(yev[m], sp[m]):.4f}")

    np.savez(args.out, coef=clf.coef_.astype(np.float32),
             intercept=clf.intercept_.astype(np.float32),
             mean=sc.mean_.astype(np.float32), scale=sc.scale_.astype(np.float32))
    print(f"TRIGGER_TRAIN_DONE cv_auc={best_auc:.4f}", flush=True)


if __name__ == "__main__":
    main()
