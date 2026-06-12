"""phaseb_labels.py — teacher-label generation for the Phase B recall indexer (PHASEB_V1).

For each synthetic conversation:
  1. Tokenize history; the last `window` tokens stay "exposed", the rest is the evicted
     region, cut into `block`-token blocks.
  2. ONE full-context prefill captures, via hooks, the PRE-RoPE k_proj outputs of the
     selected layers — i.e. CONTEXTUAL block keys (the production design harvests these
     for free during normal generation; standalone re-encoding was shown OOD by the
     Phase A probe).
  3. Teacher-force the question + gold answer with output_attentions on the answer
     tokens only (cheap: [H, T_ans, T_ctx] per layer). Label = attention mass from the
     answer tokens onto each evicted block, summed over heads and answer positions,
     with the BOS/sink position EXCLUDED before block-sum + L1 normalization (the
     label-cleaning step absent from published schemes).
  4. Save per (conversation, question): per-layer label [nB], student query features
     (pre-RoPE q at answer positions, mean-pooled -> [L, Hq, D]), block key features
     (mean+max pool over in-block contextual keys -> [nB, L, Hkv, 2D]), gold block.

Output: phaseb_labels_<shard>.npz . Run:  python3 phaseb_labels.py --n 200 --shard 0
(CPU works for smoke; GPU pod for scale.)
"""
import argparse, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

from phaseb_data import make_conversation, to_ids

LAYERS = (8, 14, 20)
WINDOW, BLOCK = 512, 128


class Capture:
    def __init__(self, llm, layers):
        self.llm, self.layers, self.k, self.q, self.hooks = llm, layers, {}, {}, []

    def __enter__(self):
        for li in self.layers:
            attn = self.llm.model.layers[li].self_attn

            def mk(store, li):
                def fn(_m, _i, o):
                    store.setdefault(li, []).append(o.detach())
                return fn
            self.hooks.append(attn.k_proj.register_forward_hook(mk(self.k, li)))
            self.hooks.append(attn.q_proj.register_forward_hook(mk(self.q, li)))
        return self

    def __exit__(self, *a):
        for h in self.hooks:
            h.remove()


@torch.no_grad()
def process(llm, tok, dev, seed):
    pairs, questions = make_conversation(seed)
    ids, _ = to_ids(tok, pairs)
    n_evict = ((len(ids) - WINDOW) // BLOCK) * BLOCK
    if n_evict < BLOCK * 4:
        return []
    nB = n_evict // BLOCK
    cfg = llm.config
    Hq, Hkv = cfg.num_attention_heads, cfg.num_key_value_heads
    D = cfg.hidden_size // Hq

    from transformers.cache_utils import DynamicCache
    out = []
    for q_text, needle_user in questions:
        gold = None
        # locate gold block: token span of the needle pair
        acc = []
        for u, a in pairs:
            text = ("<｜end▁of▁sentence｜>" if acc else "") + f"<｜User｜>{u}<｜Assistant｜>{a}"
            t = tok.encode(text, add_special_tokens=False)
            if u == needle_user:
                gold = (len(acc) + len(t) // 2) // BLOCK     # mid-pair, not pair start
            acc.extend(t)
        if gold is None or gold >= nB:
            continue
        # two-part encode so the answer offset is exact (single-string encode can shift
        # token boundaries across the join)
        pre = tok.encode(f"<｜end▁of▁sentence｜><｜User｜>{q_text}<｜Assistant｜>"
                         f"<think>\n\n</think>\n\n", add_special_tokens=False)
        ans = tok.encode(f"You told me earlier: {needle_user}", add_special_tokens=False)
        suffix, ans_start_rel = pre + ans, len(pre)
        cache = DynamicCache()
        with Capture(llm, LAYERS) as cap:
            llm.model(input_ids=torch.tensor([ids], device=dev), past_key_values=cache,
                      use_cache=True)
        kfeat = {li: cap.k[li][0][0] for li in LAYERS}            # [T_hist, Hkv*D]
        # answer pass with attentions (answer rows only would need slicing; capture all
        # suffix rows then slice). eager attn impl is set at load time.
        with Capture(llm, LAYERS) as cap2:
            o = llm.model(input_ids=torch.tensor([suffix], device=dev),
                          past_key_values=cache, use_cache=True, output_attentions=True)
        att = o.attentions                                         # L x [1,Hq,Tsuf,Tctx]
        qfeat = {li: cap2.q[li][0][0] for li in LAYERS}            # [Tsuf, Hq*D]

        labels = np.zeros((len(LAYERS), nB), dtype=np.float32)
        for lidx, li in enumerate(LAYERS):
            a = att[li][0, :, ans_start_rel:, :].float()           # [Hq, Tans, Tctx]
            mass = a.sum(dim=(0, 1))                               # [Tctx]
            mass[0] = 0.0                                          # BOS sink excluded
            ev = mass[:n_evict].view(nB, BLOCK).sum(-1)            # window/suffix excluded
            s = ev.sum().item()
            if s > 0:
                labels[lidx] = (ev / s).cpu().numpy()
        # student features
        qf = np.stack([qfeat[li][ans_start_rel:].mean(0).view(Hq, D).cpu().float().numpy()
                       for li in LAYERS])                          # [L, Hq, D]
        kf = []
        for li in LAYERS:
            k = kfeat[li][:n_evict].view(nB, BLOCK, Hkv, D).float()
            kf.append(torch.cat([k.mean(1), k.amax(1)], dim=-1).cpu().numpy())
        kf = np.stack(kf, axis=1)                                  # [nB, L, Hkv, 2D]
        out.append({"labels": labels, "q": qf.astype(np.float16),
                    "k": kf.astype(np.float16), "gold": gold, "nB": nB, "seed": seed})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--out", default="phaseb_labels_{shard}.npz")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(os.cpu_count())
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"PHASEB_LABELS_V1 dev={dev}", flush=True)
    tok = AutoTokenizer.from_pretrained("fft_hf")
    llm = AutoModelForCausalLM.from_pretrained(
        "fft_hf", dtype=torch.float32, attn_implementation="eager").eval().to(dev)
    rows = []
    for i in range(args.n):
        seed = args.shard * 100000 + i
        try:
            rows += process(llm, tok, dev, seed)
        except Exception as e:
            print(f"seed {seed} failed: {e}", flush=True)
        if (i + 1) % 10 == 0:
            print(f"{i+1}/{args.n} convs -> {len(rows)} samples", flush=True)
    pack = {}
    for j, r in enumerate(rows):
        for k, v in r.items():
            pack[f"{j}_{k}"] = v
    pack["n_samples"] = np.array(len(rows))
    np.savez_compressed(args.out.format(shard=args.shard), **pack)
    print(f"PHASEB_LABELS_DONE samples={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
