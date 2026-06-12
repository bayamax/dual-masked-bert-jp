"""probe_scores.py — diagnose BlockArchive qk scoring (BLOCKRECALL_V1).

Builds the archive from the 10-needle history and, for every needle question, ranks the
gold block (the one containing the value) under four scoring variants:

  v0 raw      : current — mean_heads -> max_keys -> mean_queries of q.k/sqrt(D)
  v1 centered : keys centered per layer/head (shared component removed) before v0
  v2 nomarker : key positions of chat-marker/special tokens dropped, then v0
  v3 softmax  : attention-style — per query/head softmax over ALL archived keys
                jointly, summed per block (mass), mean over heads/queries/layers

Hypothesis under test: high-norm boilerplate keys (chat markers) dominate max-over-keys,
flattening scores so top-k degenerates to chronological order.
"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.getcwd(), "runtime"))

import torch

from needle_recall_test import NEEDLES, build_history


def main():
    torch.set_num_threads(os.cpu_count())
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from block_recall import BlockArchive
    print("loading model ...", flush=True)
    tok = AutoTokenizer.from_pretrained("fft_hf")
    llm = AutoModelForCausalLM.from_pretrained("fft_hf", dtype=torch.float32).eval()
    hist = build_history(tok, NEEDLES)
    arc = BlockArchive(llm, tok, mode="qk")
    arc.extend(hist[:len(hist) - 512])
    L, G, D = len(arc.layers), arc.group, arc.D
    print(f"history={len(hist)} blocks={len(arc.blocks)}", flush=True)

    special = set(tok.all_special_ids or [])
    for m in ("<｜User｜>", "<｜Assistant｜>", "<｜end▁of▁sentence｜>", "<think>", "</think>"):
        special.update(tok.encode(m, add_special_tokens=False))

    # gold block per needle
    gold = {}
    for _, q, val in NEEDLES:
        for i, b in enumerate(arc.blocks):
            if val.lower() in tok.decode(b["ids"]).lower().replace(",", ""):
                gold[q] = i
                break

    # stacked keys: [L, Nk, Hkv, D], block index + marker mask per key position
    K = torch.cat([b["k"] for b in arc.blocks], dim=1).float()
    bidx = torch.tensor([i for i, b in enumerate(arc.blocks) for _ in b["ids"]])
    mark = torch.tensor([t in special for b in arc.blocks for t in b["ids"]])
    Kc = K - K.mean(dim=1, keepdim=True)
    nB = len(arc.blocks)

    def ranks(scores_fn, qc):
        out = []
        for li_idx in range(1):
            pass
        return out

    def block_scores(sim, mode):
        # sim: [L, Tq, Hq, Nk] raw dots / sqrt(D)
        if mode == "softmax":
            att = torch.softmax(sim, dim=-1)                       # over all keys jointly
            s = torch.zeros(nB)
            for i in range(nB):
                s[i] = att[..., bidx == i].sum(-1).mean()
            return s
        s = torch.full((nB,), -1e9)
        for i in range(nB):
            sub = sim[..., bidx == i]                              # [L,Tq,Hq,ki]
            s[i] = sub.mean(2).max(-1).values.mean()               # mean_h,max_k,mean_q,(mean_L)
        return s

    table = {v: [] for v in ("raw", "centered", "nomarker", "softmax")}
    for _, q, val in NEEDLES:
        qids = tok.encode(q, add_special_tokens=False)
        qc = arc._capture(qids, want_q=True)
        Q = torch.stack([qc[li].float() for li in arc.layers])     # [L,Tq,Hq,D]
        KK = K.repeat_interleave(G, 2)                             # [L,Nk,Hq,D]
        KKc = Kc.repeat_interleave(G, 2)
        sim = torch.einsum("lqhd,lkhd->lqhk", Q, KK) / D ** 0.5
        simc = torch.einsum("lqhd,lkhd->lqhk", Q, KKc) / D ** 0.5
        sim_nm = sim.clone()
        sim_nm[..., mark] = -1e9
        gi = gold.get(q, -1)
        for vname, s in (("raw", block_scores(sim, "max")),
                         ("centered", block_scores(simc, "max")),
                         ("nomarker", block_scores(sim_nm, "max")),
                         ("softmax", block_scores(sim, "softmax"))):
            order = s.argsort(descending=True).tolist()
            r = order.index(gi) if gi in order else -1
            table[vname].append(r)
        sraw = block_scores(sim, "max")
        print(f"gold={gi:2d} raw[min={sraw.min():.3f} max={sraw.max():.3f} "
              f"std={sraw.std():.4f}] ranks: " +
              " ".join(f"{v}={table[v][-1]}" for v in table), flush=True)

    print("\n=== gold-rank summary (lower=better, want <k=2) ===")
    for v, rs in table.items():
        hit2 = sum(1 for r in rs if 0 <= r < 2)
        print(f"  {v:9s} top2-hits={hit2}/10  ranks={rs}")

    # ---- per-head analysis: is the retrieval signal carried by a FEW heads? ---------
    # (retrieval-heads literature: <5% of heads do verbatim retrieval; averaging 36
    # layer-head channels would bury them. If no single head ranks gold well either,
    # the signal is absent from naive pre-RoPE QK space at this scale.)
    per_head = torch.zeros(L, arc.Hq, len(NEEDLES))
    for ni, (_, q, val) in enumerate(NEEDLES):
        qids = tok.encode(q, add_special_tokens=False)
        qc = arc._capture(qids, want_q=True)
        Q = torch.stack([qc[li].float() for li in arc.layers])
        KK = K.repeat_interleave(G, 2)
        sim = torch.einsum("lqhd,lkhd->lqhk", Q, KK) / D ** 0.5
        att = torch.softmax(sim, dim=-1)                            # [L,Tq,Hq,Nk]
        gi = gold.get(q, -1)
        for l in range(L):
            for h in range(arc.Hq):
                mass = torch.zeros(nB)
                for i in range(nB):
                    mass[i] = att[l, :, h, bidx == i].sum(-1).mean()
                order = mass.argsort(descending=True).tolist()
                per_head[l, h, ni] = order.index(gi) if gi in order else nB
    mean_rank = per_head.mean(-1)                                   # [L,Hq]
    flat = [(float(mean_rank[l, h]), l, h) for l in range(L) for h in range(arc.Hq)]
    flat.sort()
    print("\n=== best heads by mean gold-rank (softmax mass) ===")
    for mr, l, h in flat[:10]:
        hits = int((per_head[l, h] < 2).sum())
        print(f"  L{arc.layers[l]}H{h}: mean_rank={mr:.1f} top2-hits={hits}/10 "
              f"ranks={[int(r) for r in per_head[l, h]]}")
    # ensemble of the 3 best heads
    best3 = flat[:3]
    print("\n=== ensemble of best-3 heads ===")
    hits3 = []
    for ni, (_, q, val) in enumerate(NEEDLES):
        qids = tok.encode(q, add_special_tokens=False)
        qc = arc._capture(qids, want_q=True)
        Q = torch.stack([qc[li].float() for li in arc.layers])
        KK = K.repeat_interleave(G, 2)
        sim = torch.einsum("lqhd,lkhd->lqhk", Q, KK) / D ** 0.5
        att = torch.softmax(sim, dim=-1)
        mass = torch.zeros(nB)
        for _, l, h in best3:
            for i in range(nB):
                mass[i] += att[l, :, h, bidx == i].sum(-1).mean()
        order = mass.argsort(descending=True).tolist()
        gi = gold.get(q, -1)
        hits3.append(order.index(gi) if gi in order else nB)
    print(f"  ranks={hits3}  top2-hits={sum(1 for r in hits3 if r < 2)}/10")
    print("PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
