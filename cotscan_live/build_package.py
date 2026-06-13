"""build_package.py — train + SAVE the recall_kit components, build fixtures, UNIT TEST.

Produces (in OUT_DIR=recall_kit_artifacts/):
  gate.npz, indexer.npz, bge_head.npz   — loadable by recall_kit
  fixtures.pkl                          — held positives/negatives for the unit tests
  unit_report.json                      — pass/fail + metrics of the component unit tests

Reuses dolphin_scan for data; trains here so we keep the fitted models to save.
"""
import os, json, pickle, time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

import dolphin_scan as DS
from recall_kit import RecallGate, QKIndexer, BGERetriever
from recall_kit.retriever import Indexer, BGEHead

OUT = os.environ.get("OUT_DIR", "recall_kit_artifacts")
DEV = DS.DEV
LOG = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.append(s)
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "build.log"), "w").write("\n".join(LOG))


def train_gate(Xtr, ytr):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.05, max_iter=3000, class_weight="balanced").fit(
        sc.transform(Xtr), ytr)
    pos = clf.decision_function(sc.transform(Xtr))[ytr == 1]
    thresh = float(np.median(pos))
    coef = clf.coef_.ravel().astype(np.float32) / sc.scale_.astype(np.float32)
    intercept = float(clf.intercept_[0]) - float((clf.coef_.ravel() * sc.mean_ / sc.scale_).sum())
    # store in RecallGate's (coef,intercept,mean,scale,thresh) convention -> identity mean/scale
    np.savez(os.path.join(OUT, "gate.npz"),
             coef=clf.coef_.ravel().astype(np.float32),
             intercept=np.float32(clf.intercept_[0]),
             mean=sc.mean_.astype(np.float32), scale=sc.scale_.astype(np.float32),
             thresh=np.float32(thresh))
    return sc, clf, thresh


def train_indexer(rows, dims, epochs=25, seeds=2):
    Hq, Hkv, D = dims; L = len(DS.LAYERS)
    nv = max(50, int(len(rows) * 0.15)); val = rows[:nv]; tr = rows[nv:]   # held-based seed pick
    def tens(r):
        q = torch.tensor(r["q"], device=DEV)
        k = torch.tensor(r["k"], dtype=torch.float32, device=DEV)
        lab = torch.tensor(r["labels"].mean(0), device=DEV); lab = lab / max(lab.sum().item(), 1e-8)
        return q, k, lab
    best, bstate = -1, None
    for sd in range(seeds):
        torch.manual_seed(sd); m = Indexer(L, Hq, Hkv, D).to(DEV)
        opt = torch.optim.Adam(m.parameters(), lr=3e-3); rng = np.random.RandomState(sd)
        for _ in range(epochs):
            rng.shuffle(tr)
            for r in tr:
                q, k, lab = tens(r)
                loss = F.kl_div(F.log_softmax(m(q, k), 0), lab, reduction="sum")
                opt.zero_grad(); loss.backward(); opt.step()
        sc = sum(int(r["gold"]) in m(*tens(r)[:2]).topk(min(2, r["nB"])).indices.tolist()
                 for r in val) / max(len(val), 1)
        log(f"    indexer seed {sd}: val top2={sc:.3f}")
        if sc > best:
            best = sc; bstate = {k: v.detach().cpu().numpy() for k, v in m.state_dict().items()}
    np.savez(os.path.join(OUT, "indexer.npz"), **bstate)
    return bstate


def train_bge(rows, epochs=25, seeds=2):
    qdim = rows[0]["q"].reshape(-1).shape[0]; kdim = rows[0]["bge_k"].shape[1]; d = 128
    def tens(r):
        q = torch.tensor(r["q"].reshape(-1), device=DEV)
        k = torch.tensor(r["bge_k"], dtype=torch.float32, device=DEV)
        lab = torch.tensor(r["labels"].mean(0), device=DEV); lab = lab / max(lab.sum().item(), 1e-8)
        return q, k, lab
    best, bstate = -1, None
    for sd in range(seeds):
        torch.manual_seed(sd); m = BGEHead(qdim, kdim, d).to(DEV)
        opt = torch.optim.Adam(m.parameters(), lr=2e-3); rng = np.random.RandomState(sd)
        for _ in range(epochs):
            rng.shuffle(rows)
            for r in rows:
                q, k, lab = tens(r)
                loss = F.kl_div(F.log_softmax(m(q, k), 0), lab, reduction="sum")
                opt.zero_grad(); loss.backward(); opt.step()
        sc = sum(int(r["gold"]) in m(*tens(r)[:2]).topk(min(2, r["nB"])).indices.tolist()
                 for r in rows[:400]) / min(400, len(rows))
        if sc > best:
            best = sc; bstate = {k: v.detach().cpu().numpy() for k, v in m.state_dict().items()}
    np.savez(os.path.join(OUT, "bge_head.npz"), qdim=qdim, kdim=kdim, d=d, **bstate)
    return bstate


def unit_tests(fix_pos, fix_neg, dims, bge):
    """Load the SAVED weights through recall_kit and assert the components work."""
    rep = {"tests": {}, "passed": True}
    def check(name, cond, detail=""):
        rep["tests"][name] = {"pass": bool(cond), "detail": detail}
        rep["passed"] = rep["passed"] and bool(cond)
        log(f"  [unit] {name}: {'PASS' if cond else 'FAIL'} {detail}")

    gate = RecallGate.load(os.path.join(OUT, "gate.npz"))
    sp = np.mean([gate.score(r["q"].reshape(-1)) for r in fix_pos])
    sn = np.mean([gate.score(r["q"].reshape(-1)) for r in fix_neg])
    y = [1]*len(fix_pos)+[0]*len(fix_neg)
    s = [gate.score(r["q"].reshape(-1)) for r in fix_pos]+[gate.score(r["q"].reshape(-1)) for r in fix_neg]
    gate_auc = float(roc_auc_score(y, s))
    check("gate_loads_and_separates", gate_auc > 0.85, f"fixture AUC={gate_auc:.3f} (pos {sp:.2f} vs neg {sn:.2f})")

    qk = QKIndexer.load(os.path.join(OUT, "indexer.npz"), dims, device=DEV)
    hit = np.mean([int(r["gold"]) in qk.rank(r["q"], r["k"])[:2] for r in fix_pos])
    check("qk_indexer_top2", hit > 0.90, f"top-2={hit:.3f} on {len(fix_pos)} held positives")

    bge_ret = BGERetriever.load(os.path.join(OUT, "bge_head.npz"), bge, device=DEV)
    # head math on precomputed BGE embeddings (encode path covered by composite test)
    hitb = 0
    for r in fix_pos:
        q = torch.tensor(r["q"].reshape(-1), device=DEV)
        k = torch.tensor(r["bge_k"], dtype=torch.float32, device=DEV)
        order = bge_ret.head(q, k).argsort(descending=True).tolist()
        hitb += int(r["gold"]) in order[:2]
    hitb /= len(fix_pos)
    check("bge_head_top2", hitb > 0.85, f"top-2={hitb:.3f} (learned transform on BGE)")

    # encode path smoke: BGE actually embeds text and ranks
    sample = fix_pos[0]
    enc = bge_ret.bge.encode(["the price was 6800 yen", "the weather is nice"],
                             normalize_embeddings=True, show_progress_bar=False)
    check("bge_encode_path", enc.shape[0] == 2 and enc.shape[1] == sample["bge_k"].shape[1],
          f"BGE encodes to {enc.shape}")
    json.dump(rep, open(os.path.join(OUT, "unit_report.json"), "w"), indent=2)
    return rep


def main():
    t0 = time.time(); os.makedirs(OUT, exist_ok=True)
    tok, llm, pooler, embT, bge = DS.build_model()
    cfg = llm.config; dims = (cfg.num_attention_heads, cfg.num_key_value_heads,
                              cfg.hidden_size // cfg.num_attention_heads)
    N_TR = int(os.environ.get("PKG_N_TRAIN", "300")); N_HE = int(os.environ.get("PKG_N_HELD", "60"))
    N_CO = int(os.environ.get("PKG_N_COMP", "60"))
    pool = DS.pick_items(N_TR + N_HE + N_CO)
    tr_pool, te_pool, co_pool = pool[:N_TR], pool[N_TR:N_TR + N_HE], pool[N_TR + N_HE:]
    # hand the composite test items from the SAME sorted pool (same CoT-length regime)
    json.dump([[u, c] for _, u, c in co_pool],
              open(os.path.join(OUT, "composite_items.json"), "w"))
    log(f"[data] train_items={len(tr_pool)} held_items={len(te_pool)} comp_items={len(co_pool)}")
    Xtr, ytr, _, IXtr = DS.collect(tr_pool, llm, tok, pooler, embT, dims, "TRAIN", t0, bge=bge)
    Xte, yte, _, IXte = DS.collect(te_pool, llm, tok, pooler, embT, dims, "HELD", t0, bge=bge)
    log(f"[data] gate train {len(ytr)}({int(ytr.sum())}+) held {len(yte)}({int(yte.sum())}+) "
        f"idx_pos train {len(IXtr)} held {len(IXte)}")

    log("[train] gate ..."); train_gate(Xtr, ytr)
    log("[train] indexer ..."); train_indexer([dict(r) for r in IXtr], dims)
    log("[train] bge head ..."); train_bge([dict(r) for r in IXtr])

    # fixtures: held positives (full info) + held-negative gate features
    fix_pos = [{"q": r["q"], "k": r["k"], "bge_k": r["bge_k"], "gold": int(r["gold"]),
                "nB": int(r["nB"])} for r in IXte[:120]]
    negq = Xte[yte == 0][:120]
    fix_neg = [{"q": q.reshape(len(DS.LAYERS), dims[0], dims[2])} for q in negq]
    pickle.dump({"pos": fix_pos, "neg": fix_neg, "dims": dims},
                open(os.path.join(OUT, "fixtures.pkl"), "wb"))
    log(f"[fixtures] saved {len(fix_pos)} pos + {len(fix_neg)} neg")

    log("[unit tests] running ...")
    rep = unit_tests(fix_pos, fix_neg, dims, bge)
    log(f"[unit tests] {'ALL PASS' if rep['passed'] else 'FAILURES'} t={time.time()-t0:.0f}s")
    open(os.path.join(OUT, ".build_done"), "w").write("ok")


if __name__ == "__main__":
    main()
