"""attn_probe4 — retrain the retrieval trigger IN THE PRODUCTION CONTEXT LAYOUT
(MULTITURN_TRIGGER_REPORT.md: the v3 trigger over-fires at 0.9-1.0 on everything).

Why v3 saturates: it was trained on [BOS + SP(single turn-1 fact) + question] — no raw
window, SP compressing ~30 tokens. Production scores turns in
[SP(whole distant buffer) + recent raw window + question]: both the SP content
distribution and the attention competition are different worlds, so the standardisation
(mu/sd) maps production masses to huge z-scores and the sigmoid pins at ~1.0.

This probe rebuilds the dataset in the production shape:
  * a multi-turn conversation prefix (stitched User/Assistant turns) ~700+ tokens
  * the scenario's turn-1 fact embedded EARLY so it lands in the DISTANT (kept) region
  * kept = prefix[:-rw] compressed by the pooler -> SP;  recent = prefix[-rw:] raw
  * turn-2 = referential (needs the distant value -> must reach into SP) or control twin
Features: per-head attention mass from turn-2 positions onto SP positions, full grid;
head re-selection INSIDE each LOO fold (heads may differ in-context); saves
attn_trigger4_head.npz (numpy-only, heads + coef + mu/sd) and an operating table.

Run next to fft_hf/ and fft_out/pooler.pt:  python3 attn_probe4.py [--rw 256]
"""
import argparse, os, sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attn_export3_torch import load_pooler
from attn_scenarios import SCEN

# neutral filler turns (no numbers — they must not compete as 'the value')
FILLER = [
    ("Tell me something interesting about lighthouses.",
     "Lighthouses once burned coal and wood before lenses concentrated lamp light, and keepers lived on site year round."),
    ("What makes sourdough different from regular bread?",
     "Sourdough rises with wild yeast and lactic bacteria from a fermented starter rather than commercial yeast, giving a tangier crumb."),
    ("Any tips for keeping houseplants healthy?",
     "Match each plant to its light, let soil dry between waterings, and repot when roots circle the container."),
    ("How do noise-cancelling headphones work?",
     "Microphones sample ambient sound and the speakers emit an inverted waveform that cancels much of the steady noise."),
    ("Why do cats knead with their paws?",
     "Kneading is a kitten nursing behaviour carried into adulthood, signalling comfort and marking with scent glands."),
    ("What's a good way to start journaling?",
     "Keep the bar low: a few honest lines daily at the same time, ideally tied to an existing habit like morning coffee."),
]


def build_prefix(tok, t1_fact, min_distant, rw):
    """Conversation prefix whose token length puts t1 (placed FIRST) inside the distant
    region by at least `min_distant` tokens of margin."""
    parts = [f"<｜User｜>{t1_fact}<｜Assistant｜>Got it — noted."]
    i = 0
    ids = tok.encode("".join(parts), add_special_tokens=False)
    while len(ids) < min_distant + rw:
        q, a = FILLER[i % len(FILLER)]
        parts.append(f"<｜end▁of▁sentence｜><｜User｜>{q}<｜Assistant｜>{a}")
        i += 1
        ids = tok.encode("".join(parts), add_special_tokens=False)
    return ids


@torch.no_grad()
def masses(llm, embT, pooler, tok, prefix_ids, rw, t2_text):
    kept, recent = prefix_ids[:-rw], prefix_ids[-rw:]
    sp = pooler(embT(torch.tensor([kept])).float())
    n_sp = sp.shape[1]
    t2 = tok.encode(f"<｜end▁of▁sentence｜><｜User｜>{t2_text}<｜Assistant｜>",
                    add_special_tokens=False)
    inp = torch.cat([embT(torch.tensor([[tok.bos_token_id]])), sp,
                     embT(torch.tensor([recent + t2]))], 1)
    off_sp = 1
    off_q = 1 + n_sp + len(recent)                 # turn-2 query positions
    out = llm(inputs_embeds=inp, output_attentions=True, use_cache=False)
    L, H = len(out.attentions), out.attentions[0].shape[1]
    M = np.zeros((L, H))
    for li, A in enumerate(out.attentions):
        q = A[0][:, off_q:, :]
        M[li] = q[:, :, off_sp:off_sp + n_sp].sum(-1).mean(-1).numpy()
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rw", type=int, default=256)
    ap.add_argument("--min-distant", type=int, default=350)
    a = ap.parse_args()
    torch.set_num_threads(os.cpu_count())
    tok = AutoTokenizer.from_pretrained("fft_hf")
    llm = AutoModelForCausalLM.from_pretrained("fft_hf", dtype=torch.float32,
                                               attn_implementation="eager").eval()
    embT = llm.get_input_embeddings()
    pooler = load_pooler()

    print(f"collecting production-layout masses (rw={a.rw}, distant>={a.min_distant})...",
          flush=True)
    refM, ctrlM, cats = [], [], []
    for i, (cat, t1, ref, ctrl) in enumerate(SCEN):
        prefix = build_prefix(tok, t1, a.min_distant, a.rw)
        refM.append(masses(llm, embT, pooler, tok, prefix, a.rw, ref))
        ctrlM.append(masses(llm, embT, pooler, tok, prefix, a.rw, ctrl))
        cats.append(cat)
        print(f"  [{i + 1}/{len(SCEN)}] {cat} prefix={len(prefix)} tok", flush=True)
    refM, ctrlM = np.array(refM), np.array(ctrlM)
    n, L, H = refM.shape
    print(f"mean SP-mass: ref={refM.mean():.4f} ctrl={ctrlM.mean():.4f} "
          f"delta={(refM - ctrlM).mean():+.4f}", flush=True)

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    TOPK = 6

    def topk(tr):
        D = (refM[tr] - ctrlM[tr]).mean(0)
        return [(l, h) for _, l, h in
                sorted(((D[l, h], l, h) for l in range(L) for h in range(H)), reverse=True)[:TOPK]]

    def feats(M, heads):
        return np.array([M[l, h] for l, h in heads])

    probs, ys, pair, sel = [], [], 0, {}
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        heads = topk(tr)
        for hd in heads:
            sel[hd] = sel.get(hd, 0) + 1
        Xtr = np.array([feats(refM[j], heads) for j in tr] + [feats(ctrlM[j], heads) for j in tr])
        ytr = np.array([1] * len(tr) + [0] * len(tr))
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
        clf = LogisticRegression(max_iter=2000, C=1.0).fit((Xtr - mu) / sd, ytr)
        pr = clf.predict_proba((np.stack([feats(refM[i], heads), feats(ctrlM[i], heads)]) - mu) / sd)[:, 1]
        probs += [float(pr[0]), float(pr[1])]; ys += [1, 0]; pair += int(pr[0] > pr[1])
    auc = roc_auc_score(ys, probs)
    print(f"\n== production-layout LOO-CV ==\n  AUC={auc:.3f} pairwise={pair}/{n}")
    print("  stable heads:", sorted(sel.items(), key=lambda kv: -kv[1])[:8])

    heads = topk(list(range(n)))
    X = np.array([feats(refM[j], heads) for j in range(n)] + [feats(ctrlM[j], heads) for j in range(n)])
    y = np.array([1] * n + [0] * n)
    mu, sd = X.mean(0), X.std(0) + 1e-8
    clf = LogisticRegression(max_iter=2000, C=1.0).fit((X - mu) / sd, y)
    pr = clf.predict_proba((X - mu) / sd)[:, 1]
    print("\n== operating table (in-sample, threshold selection) ==")
    print("  th    recall  FPR")
    for th in (0.3, 0.4, 0.5, 0.6, 0.7):
        tp = int((pr[:n] >= th).sum()); fp = int((pr[n:] >= th).sum())
        print(f"  {th:.1f}   {tp}/{n}    {fp}/{n}")
    np.savez("attn_trigger4_head.npz", coef=clf.coef_, intercept=clf.intercept_,
             classes=np.array(["ctrl", "ref"], dtype=object), mu=mu, sd=sd,
             heads=np.array(heads), rw=np.array([a.rw]), loocv_auc=np.array([auc]))
    print(f"\nsaved attn_trigger4_head.npz heads={[f'L{l}H{h}' for l, h in heads]}")
    print("ATTN_PROBE4_DONE")


if __name__ == "__main__":
    main()
