"""STEP2 (torch), scaled (STATUS 2026-06-10, next-step 1): turn the per-head SP-attention
signal into an actual RETRIEVAL TRIGGER — a logistic classifier over the per-head
attention-to-SP masses that predicts "this turn needs a value that lives only in the SP"
(referential) vs "this turn is self-contained" (control).

Honest protocol (the 5-example probe couldn't have this):
  * leave-one-SCENARIO-out CV — both members of a ref/ctrl pair are held out together, so
    a fold never sees its own scenario;
  * head selection (top-k by ref-ctrl delta) re-done INSIDE every training fold — the
    headline AUC therefore contains no selection leakage;
  * head-stability report — how often each head is selected across folds (a trigger built
    on heads that only win on one task family would not deploy well).

Saves attn_trigger3.joblib {heads, clf, mu, sd} fit on the full data for runtime use:
score a turn by running one eager-attention forward, reading the selected heads' SP mass,
and firing external (L1/L2/L3) retrieval when the probability clears the threshold —
replacing the heuristic "is this referential?" routing with the model's own signal.

Run after attn_export3.py, next to fft_hf/:  python3 attn_probe3.py
"""
import numpy as np, torch
from transformers import AutoModelForCausalLM

TOPK = 6                      # heads fed to the logistic trigger
d = np.load("attn_probe3.npz", allow_pickle=True)
n = int(d["n"][0]); bos = int(d["bos"][0]); cat = [str(c) for c in d["cat"]]
llm = AutoModelForCausalLM.from_pretrained("fft_hf", dtype=torch.float32, attn_implementation="eager").eval()
embT = llm.get_input_embeddings()


def per_head_sp_attn(sp, t2ids):
    """(L,H) matrix: for each layer/head, mean over turn-2 query positions of attention mass on SP."""
    n_sp = sp.shape[0]; off = 1 + n_sp
    inp = torch.cat([embT(torch.tensor([bos])), torch.tensor(sp, dtype=torch.float32),
                     embT(torch.tensor(t2ids, dtype=torch.long))], 0)[None]
    with torch.no_grad():
        out = llm(inputs_embeds=inp, output_attentions=True, use_cache=False)
    L = len(out.attentions); Hh = out.attentions[0].shape[1]
    M = np.zeros((L, Hh))
    for li, A in enumerate(out.attentions):
        q = A[0][:, off:, :]
        M[li] = q[:, :, 1:1 + n_sp].sum(-1).mean(-1).numpy()
    return M


print("collecting per-head masses...")
refMs = np.array([per_head_sp_attn(d[f"sp_{i}"], d[f"ref_{i}"]) for i in range(n)])    # (n,L,H)
ctrlMs = np.array([per_head_sp_attn(d[f"sp_{i}"], d[f"ctrl_{i}"]) for i in range(n)])
L, Hh = refMs.shape[1:]
print(f"layers={L} heads={Hh} | scenarios={n} | mean ref={refMs.mean():.3f} ctrl={ctrlMs.mean():.3f} "
      f"delta={(refMs - ctrlMs).mean():+.3f}")

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def topk_heads(tr_idx, k=TOPK):
    D = (refMs[tr_idx] - ctrlMs[tr_idx]).mean(0)
    flat = sorted(((D[l, h], l, h) for l in range(L) for h in range(Hh)), reverse=True)
    return [(l, h) for _, l, h in flat[:k]]


def feats(M_lh, heads):
    return np.array([M_lh[l, h] for l, h in heads])


# ---- leave-one-scenario-out CV (selection + standardisation re-fit per fold) ----------
probs, ys, sel_count = [], [], {}
pair_hits = 0
for i in range(n):
    tr = [j for j in range(n) if j != i]
    heads = topk_heads(tr)
    for hd in heads:
        sel_count[hd] = sel_count.get(hd, 0) + 1
    Xtr = np.array([feats(refMs[j], heads) for j in tr] + [feats(ctrlMs[j], heads) for j in tr])
    ytr = np.array([1] * len(tr) + [0] * len(tr))
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    clf = LogisticRegression(max_iter=2000, C=1.0).fit((Xtr - mu) / sd, ytr)
    Xte = (np.stack([feats(refMs[i], heads), feats(ctrlMs[i], heads)]) - mu) / sd
    pr = clf.predict_proba(Xte)[:, 1]
    probs += [float(pr[0]), float(pr[1])]; ys += [1, 0]
    pair_hits += int(pr[0] > pr[1])

probs, ys = np.array(probs), np.array(ys)
auc = roc_auc_score(ys, probs)
acc05 = float(((probs >= 0.5).astype(int) == ys).mean())
print(f"\n== leave-one-scenario-out CV (no selection leakage) ==")
print(f"  AUC = {auc:.3f} | accuracy@0.5 = {acc05:.3f} | pairwise (ref>ctrl) = {pair_hits}/{n}")

print(f"\n== per-category pairwise accuracy ==")
for c in dict.fromkeys(cat):
    idx = [i for i in range(n) if cat[i] == c]
    hits = sum(int(probs[2 * i] > probs[2 * i + 1]) for i in idx)
    print(f"  {c:>5}: {hits}/{len(idx)}")

print(f"\n== head-selection stability across {n} folds (selected-in-fold count) ==")
for (l, h), c in sorted(sel_count.items(), key=lambda kv: -kv[1])[:12]:
    D = (refMs - ctrlMs).mean(0)
    print(f"  L{l:>2}H{h:>2}: {c:>3}/{n}  full-data delta={D[l, h]:+.3f}")

# ---- final trigger on the full data ----------------------------------------------------
heads = topk_heads(list(range(n)))
X = np.array([feats(refMs[j], heads) for j in range(n)] + [feats(ctrlMs[j], heads) for j in range(n)])
y = np.array([1] * n + [0] * n)
mu, sd = X.mean(0), X.std(0) + 1e-8
clf = LogisticRegression(max_iter=2000, C=1.0).fit((X - mu) / sd, y)
import joblib
joblib.dump({"heads": heads, "clf": clf, "mu": mu, "sd": sd,
             "loocv_auc": float(auc), "n_scenarios": n}, "attn_trigger3.joblib")
print(f"\nsaved attn_trigger3.joblib | heads={[f'L{l}H{h}' for l, h in heads]}")
print("ATTN_PROBE3_DONE")
