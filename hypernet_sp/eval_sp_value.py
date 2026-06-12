"""SP value A/B (release-quality gate): does carrying compressed gist beat just DROPPING
distant context? Four context treatments are scored on the SAME prediction targets, on
held-out in-distribution data (dolphin-r1 reasoning-deepseek, reasoning 1500-3500 tokens —
DISJOINT from training by construction: train_ce/prep_long filtered to >=4000).

Transcript layout per sample (token budgets via --distant/--rw/--eval):
    [query] [distant ...] [recent rw] [eval targets]

Treatments (context the model sees when predicting the eval targets):
  full     [query][distant][recent]          — upper bound (unbounded KV)
  sp       [query][SP(distant)][recent]      — the shipping runtime (bounded)
  drop     [query][recent]                   — R1 chat template behaviour (past think deleted)
  rand-sp  [query][noise-SP][recent]         — placebo: is it the CONTENT of the SP or just
                                               32 extra vectors of slack?

Reports mean CE on the eval targets per treatment, the recovery ratio
(drop-sp)/(drop-full) — how much of the full-attention gap the SP recovers — and per-sample
win rates. Run next to fft_hf/ and fft_out/pooler.pt:
    python3 eval_sp_value.py [--n 20] [--distant 512] [--rw 128] [--eval 96]
"""
import argparse, json, os, sys, time
import numpy as np, torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attn_export3_torch import load_pooler


def get_samples(tok, n, min_t=1500, max_t=3500, need=None):
    cache = "sp_value_samples.json"
    if os.path.exists(cache):
        return json.load(open(cache))[:n]
    from datasets import load_dataset
    ds = load_dataset("cognitivecomputations/dolphin-r1", "reasoning-deepseek",
                      split="train", streaming=True)
    need = need or n
    out = []
    for r in ds:
        rsn = r.get("reasoning") or ""
        if not (min_t * 3 <= len(rsn) <= max_t * 5):           # cheap char prefilter
            continue
        m = r.get("messages"); q = None
        if isinstance(m, list):
            for t in m:
                if (t.get("role") or "").lower() == "user":
                    q = t.get("content"); break
        if not q or not (20 <= len(q) <= 700):
            continue
        ids = tok(rsn, add_special_tokens=False).input_ids
        if not (min_t <= len(ids) <= max_t):
            continue
        out.append({"q": q, "rsn_ids": ids})
        if len(out) >= need:
            break
    json.dump(out, open(cache, "w"))
    return out[:n]


@torch.no_grad()
def ce_on_targets(llm, ctx_emb, tgt_ids):
    """CE of tgt_ids given a context provided as embeddings (the SP slots have no ids)."""
    tgt = torch.tensor([tgt_ids])
    inp = torch.cat([ctx_emb, llm.get_input_embeddings()(tgt)], 1)
    logits = llm(inputs_embeds=inp).logits[0]
    T = len(tgt_ids)
    pred = logits[-T - 1:-1]                                    # positions predicting each target
    return float(F.cross_entropy(pred, tgt[0], reduction="mean"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--distant", type=int, default=512)
    ap.add_argument("--rw", type=int, default=128)
    ap.add_argument("--eval", type=int, default=96, dest="ev")
    a = ap.parse_args()
    torch.set_num_threads(os.cpu_count())

    tok = AutoTokenizer.from_pretrained("fft_hf")
    llm = AutoModelForCausalLM.from_pretrained("fft_hf", dtype=torch.float32).eval()
    embT = llm.get_input_embeddings()
    pooler = load_pooler()
    need = a.distant + a.rw + a.ev
    samples = [s for s in get_samples(tok, a.n * 2) if len(s["rsn_ids"]) >= need][:a.n]
    print(f"{len(samples)} held-out samples (reasoning >= {need} tok) | "
          f"distant={a.distant} rw={a.rw} eval={a.ev}")

    res = {k: [] for k in ("full", "sp", "drop", "rand")}
    g = torch.Generator().manual_seed(0)
    for i, s in enumerate(samples):
        q_ids = tok(f"<｜User｜>{s['q']}<｜Assistant｜>", add_special_tokens=True).input_ids
        r = s["rsn_ids"]
        distant, recent, tgt = r[:a.distant], r[a.distant:a.distant + a.rw], \
            r[a.distant + a.rw:a.distant + a.rw + a.ev]
        qe, de, re_ = embT(torch.tensor([q_ids])), embT(torch.tensor([distant])), \
            embT(torch.tensor([recent]))
        sp = pooler(de.float()).to(qe.dtype)
        rand = torch.randn(sp.shape, generator=g)
        rand = rand / rand.norm(dim=-1, keepdim=True) * sp.norm(dim=-1, keepdim=True)  # placebo @ same norm
        t0 = time.time()
        res["full"].append(ce_on_targets(llm, torch.cat([qe, de, re_], 1), tgt))
        res["sp"].append(ce_on_targets(llm, torch.cat([qe, sp, re_], 1), tgt))
        res["drop"].append(ce_on_targets(llm, torch.cat([qe, re_], 1), tgt))
        res["rand"].append(ce_on_targets(llm, torch.cat([qe, rand, re_], 1), tgt))
        print(f"  [{i + 1}/{len(samples)}] full={res['full'][-1]:.3f} sp={res['sp'][-1]:.3f} "
              f"drop={res['drop'][-1]:.3f} rand={res['rand'][-1]:.3f}  ({time.time() - t0:.0f}s)",
              flush=True)

    M = {k: float(np.mean(v)) for k, v in res.items()}
    n = len(samples)
    sp_w_drop = sum(s < d for s, d in zip(res["sp"], res["drop"]))
    sp_w_rand = sum(s < r for s, r in zip(res["sp"], res["rand"]))
    rec = (M["drop"] - M["sp"]) / max(M["drop"] - M["full"], 1e-9)
    print("\n== SP value A/B (mean CE on eval targets; lower better) ==")
    for k, label in (("full", "full attention (upper bound)"), ("sp", "SP + recent (shipping)"),
                     ("drop", "recent only (drop, R1 template)"), ("rand", "noise-SP + recent (placebo)")):
        print(f"  {k:>5}: {M[k]:.4f}   {label}")
    print(f"\n  gap to full:   sp +{M['sp'] - M['full']:.4f} | drop +{M['drop'] - M['full']:.4f}")
    print(f"  SP recovers {rec:.1%} of the drop->full gap")
    print(f"  per-sample wins: sp<drop {sp_w_drop}/{n} | sp<rand {sp_w_rand}/{n}")
    json.dump({"mean": M, "per_sample": res,
               "cfg": {"distant": a.distant, "rw": a.rw, "eval": a.ev}},
              open("sp_value_results.json", "w"), indent=1)
    print("saved sp_value_results.json")
    print("SP_VALUE_DONE")


if __name__ == "__main__":
    main()
