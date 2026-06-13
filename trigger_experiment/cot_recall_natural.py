"""cot_recall_natural.py — COTRECALL natural arm: the model writes its OWN long CoT.

The teacher-forced arm (cot_recall_labels.py) puts the fire position at the end of a
fixed lead. This arm instead lets R1-distill reason freely under full-KV (the value IS
visible in history) with a REAL budget (--maxgen 4000; R1 thinks long, the 90-token cap
was why the first attempt yielded <5%), and takes the position where it first naturally
emits the value as the fire position. Then harvests the production-context query there
(q_fire) vs over the oblique question (q_question), same keys/labels, for the indexer
comparison — the faithful test of "use the mid-CoT hidden state as the retrieval query".

Output: cot_recall_natural_<shard>.npz . Run: python3 cot_recall_natural.py --n 100 --shard 0
"""
import argparse, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

from attn_export3_torch import load_pooler
from cot_recall_labels import (BLOCK, LAYERS, WINDOW, build_conversation, to_ids)
from recall_query_labels import Capture


@torch.no_grad()
def gen_until_value(llm, tok, dev, prompt_ids, value, maxgen):
    """Greedy full-KV generation until `value` appears. Returns (cot_ids_before_value,
    n_generated) or (None, n_generated) if never reached within maxgen."""
    from transformers.cache_utils import DynamicCache
    cache = DynamicCache()
    out = llm(input_ids=torch.tensor([prompt_ids], device=dev),
              past_key_values=cache, use_cache=True)
    nxt = int(out.logits[0, -1].argmax())
    gen, norm = [], value.replace(",", "")
    for step in range(maxgen):
        gen.append(nxt)
        if norm in tok.decode(gen).replace(",", ""):
            m = len(gen)
            while m > 0 and norm in tok.decode(gen[:m]).replace(",", ""):
                m -= 1
            return gen[:m], step + 1
        out = llm(input_ids=torch.tensor([[nxt]], device=dev),
                  past_key_values=cache, use_cache=True)
        nxt = int(out.logits[0, -1].argmax())
    return None, maxgen


@torch.no_grad()
def process(llm, tok, pooler, dev, seed, maxgen, held=False):
    from cot_recall_labels import ITEMS
    split = len(ITEMS) * 3 // 4                              # template-disjoint heldout
    pool = list(range(split, len(ITEMS))) if held else list(range(split))
    item_idx = pool[seed % len(pool)]
    built = build_conversation(seed, item_idx)
    if built is None:
        return []
    pairs, needle_user, q_text, _lead, value = built
    ids, spans = to_ids(tok, pairs)
    n_evict = ((len(ids) - WINDOW) // BLOCK) * BLOCK
    if n_evict < BLOCK * 4:
        return []
    nB = n_evict // BLOCK
    gold = None
    for i, (u, _a) in enumerate(pairs):
        if u == needle_user:
            s, e = spans[i]
            gold = (s + (e - s) // 2) // BLOCK
    if gold is None or gold >= nB:
        return []
    cfg = llm.config
    Hq, Hkv = cfg.num_attention_heads, cfg.num_key_value_heads
    D = cfg.hidden_size // Hq
    embT = llm.get_input_embeddings()
    bos = tok.bos_token_id

    pre = tok.encode(f"<｜end▁of▁sentence｜><｜User｜>{q_text}<｜Assistant｜><think>\n",
                     add_special_tokens=False)
    cot_ids, n_gen = gen_until_value(llm, tok, dev, ids + pre, value, maxgen)
    if not cot_ids:
        return []                                            # model never reached value
    val_ids = tok.encode(value, add_special_tokens=False)
    suffix = pre + cot_ids + val_ids
    fire_pos = len(pre) + len(cot_ids) - 1
    val_start = len(pre) + len(cot_ids)

    from transformers.cache_utils import DynamicCache
    cache = DynamicCache()
    with Capture(llm, LAYERS, want_q=False, want_k=True) as cap:
        llm.model(input_ids=torch.tensor([ids], device=dev),
                  past_key_values=cache, use_cache=True)
    kf = []
    for li in LAYERS:
        k = cap.k[li][0][0][:n_evict].view(nB, BLOCK, Hkv, D).float()
        kf.append(torch.cat([k.mean(1), k.amax(1)], dim=-1).cpu().numpy())
    kf = np.stack(kf, axis=1).astype(np.float16)
    o = llm.model(input_ids=torch.tensor([suffix], device=dev),
                  past_key_values=cache, use_cache=True, output_attentions=True)
    att = o.attentions
    labels = np.zeros((len(LAYERS), nB), dtype=np.float32)
    for lidx, li in enumerate(LAYERS):
        a = att[li][0].float()
        m = a[:, val_start:, :].sum(dim=(0, 1)); m[0] = 0.0
        ev = m[:n_evict].view(nB, BLOCK).sum(-1)
        if ev.sum().item() > 0:
            labels[lidx] = (ev / ev.sum()).cpu().numpy()

    kept, window = ids[:n_evict], ids[n_evict:]
    sp = pooler(embT(torch.tensor([kept], device=dev)).float()).to(embT.weight.dtype)
    win_emb = embT(torch.tensor([window], device=dev))
    c = DynamicCache()
    llm.model(input_ids=torch.tensor([[bos]], device=dev), past_key_values=c, use_cache=True)
    llm.model(inputs_embeds=torch.cat([sp, win_emb], 1), past_key_values=c, use_cache=True)
    with Capture(llm, LAYERS, want_q=True, want_k=False) as cap2:
        llm.model(inputs_embeds=embT(torch.tensor([suffix], device=dev)),
                  past_key_values=c, use_cache=True)
    qsp = cap2.q

    def pick(a, b=None):
        return np.stack([qsp[li][0][0][a:b].mean(0).view(Hq, D).cpu().float().numpy()
                         for li in LAYERS]).astype(np.float16)

    return [{"labels": labels, "k": kf, "gold": np.array(gold), "nB": np.array(nB),
             "q_question": pick(0, len(pre)), "q_fire": pick(fire_pos, fire_pos + 1),
             "q_fire_generic": pick(fire_pos, fire_pos + 1),   # same (no generic lead here)
             "q_ctrl": pick(len(pre), len(pre) + 1),
             "fire_mass": np.array(0.0, dtype=np.float32),
             "ctrl_mass": np.array(0.0, dtype=np.float32),
             "n_gen": np.array(n_gen), "seed": np.array(seed)}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--maxgen", type=int, default=4000)
    ap.add_argument("--held", action="store_true")
    ap.add_argument("--out", default="cot_recall_natural_{shard}.npz")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(os.cpu_count())
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"COTRECALL_NATURAL dev={dev} maxgen={args.maxgen}", flush=True)
    tok = AutoTokenizer.from_pretrained("fft_hf")
    llm = AutoModelForCausalLM.from_pretrained(
        "fft_hf", dtype=torch.float32, attn_implementation="eager").eval().to(dev)
    pooler = load_pooler().to(dev)
    rows, reached, gens = [], 0, []
    for i in range(args.n):
        seed = args.shard * 100000 + i
        try:
            r = process(llm, tok, pooler, dev, seed, args.maxgen, held=args.held)
            rows += r
            if r:
                reached += 1; gens.append(int(r[0]["n_gen"]))
        except Exception as e:
            print(f"seed {seed} failed: {e}", flush=True)
        if (i + 1) % 5 == 0:
            mg = int(np.mean(gens)) if gens else 0
            print(f"{i+1}/{args.n} reached={reached} avg_gen_tokens={mg}", flush=True)
    pack = {}
    for j, r in enumerate(rows):
        for k, v in r.items():
            pack[f"{j}_{k}"] = v
    pack["n_samples"] = np.array(len(rows))
    np.savez_compressed(args.out.format(shard=args.shard), **pack)
    yld = reached / max(args.n, 1)
    print(f"COTRECALL_NATURAL_DONE samples={len(rows)} reached={reached} yield={yld:.2f} "
          f"avg_gen={int(np.mean(gens)) if gens else 0}", flush=True)


if __name__ == "__main__":
    main()
