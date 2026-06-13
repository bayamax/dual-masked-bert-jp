"""cot_attn_scan_labels.py — COTSCAN_V1: attention-defined recall position.

The faithful version of the user's spec. No generation. For each (evicted fact,
oblique compute question):

  1. Teacher-force a short compute answer that USES the value.
  2. Full-KV: at EVERY answer-token position, measure attention mass to the GOLD BLOCK
     (the fact, >512 tokens back). The recall moment is DEFINED by this signal — the
     position where attention to the gold block peaks — not by any hand-picked token.
  3. Production context [BOS][SP][window]: harvest the pre-RoPE query (hidden rep) at
     that attention-defined recall position, at a non-recall answer position, and over
     the question.

Then cot_recall_eval.py (reused, same field names) answers:
  Q1 retrieval: from the recall-position hidden rep, can we pull back the gold block?
     (vs the question-position hidden rep)  -> q_fire vs q_question top-2.
  Q2 per-position gate: do attention to gold (fire_mass vs ctrl_mass) and the SP-context
     hidden rep separate recall from non-recall positions? -> probe AUC.

Output: cot_recall_labels_<shard>.npz (same schema). Run: python3 cot_attn_scan_labels.py --n 200 --shard 0
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

from attn_export3_torch import load_pooler
from cot_recall_labels import (BLOCK, ITEMS, LAYERS, WINDOW, build_conversation, to_ids)
from recall_query_labels import Capture


def product(value, mult):
    try:
        if "." in value:
            return str(round(float(value) * mult, 1))
        return str(int(value) * mult)
    except ValueError:
        return value


@torch.no_grad()
def process(llm, tok, pooler, dev, seed, held=False):
    split = len(ITEMS) * 3 // 4
    pool = list(range(split, len(ITEMS))) if held else list(range(split))
    item_idx = pool[seed % len(pool)]
    built = build_conversation(seed, item_idx)
    if built is None:
        return []
    pairs, needle_user, q_text, _lead, value = built
    mult = ITEMS[item_idx].get("mult", 2)
    ids, spans = to_ids(tok, pairs)
    n_evict = ((len(ids) - WINDOW) // BLOCK) * BLOCK
    if n_evict < BLOCK * 4:
        return []
    nB = n_evict // BLOCK
    gold, gspan = None, None
    for i, (u, _a) in enumerate(pairs):
        if u == needle_user:
            s, e = spans[i]
            gold = (s + (e - s) // 2) // BLOCK
            gspan = (gold * BLOCK, (gold + 1) * BLOCK)
    if gold is None or gold >= nB:
        return []
    cfg = llm.config
    Hq, Hkv = cfg.num_attention_heads, cfg.num_key_value_heads
    D = cfg.hidden_size // Hq
    embT = llm.get_input_embeddings()
    bos = tok.bos_token_id

    pre = tok.encode(f"<｜end▁of▁sentence｜><｜User｜>{q_text}<｜Assistant｜>",
                     add_special_tokens=False)
    answer = f"Working it out: that figure is {value}, so {value} times {mult} is " \
             f"{product(value, mult)}."
    ans_ids = tok.encode(answer, add_special_tokens=False)
    suffix = pre + ans_ids
    a0, a1 = len(pre), len(pre) + len(ans_ids)              # answer span

    from transformers.cache_utils import DynamicCache
    # full-KV: block keys + per-answer-position attention to the gold block ------------
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
    # per answer position: attention mass to each evicted block (sum heads+layers,
    # BOS excluded, row-normalized). This is the teacher signal that defines BOTH the
    # recall position AND the target block.
    blockmass = np.zeros((len(ans_ids), nB), dtype=np.float32)
    to_evict = np.zeros(len(ans_ids), dtype=np.float32)
    for li in LAYERS:
        a = att[li][0].float()                              # [Hq, T, Tctx]
        rows = a[:, a0:a1, :].sum(0)                        # [Lans, Tctx]
        rows[:, 0] = 0.0
        denom = rows.sum(-1, keepdim=True).clamp(min=1e-8)
        to_evict += (rows[:, :n_evict].sum(-1) / denom[:, 0]).cpu().numpy()
        bm = rows[:, :n_evict].view(len(ans_ids), nB, BLOCK).sum(-1)
        blockmass += (bm / denom).cpu().numpy()
    blockmass /= len(LAYERS); to_evict /= len(LAYERS)
    # recall position = where attention concentrates most on a SINGLE far block
    # (= "a token >512 back gets attention above all others"); gold = that block.
    peak_per_pos = blockmass.max(axis=1)                    # [Lans]
    recall_rel = int(peak_per_pos.argmax())
    nonrecall_rel = int(peak_per_pos.argmin())
    gold = int(blockmass[recall_rel].argmax())             # ATTENTION-defined gold block
    gold_fact = (gspan[0] // BLOCK)                         # planted-fact block (cross-check)
    # retrieval label = block attention profile AT the recall position
    lab = blockmass[recall_rel]
    labels = np.tile((lab / max(lab.sum(), 1e-8))[None], (len(LAYERS), 1)).astype(np.float32)

    # production context: pre-RoPE query at recall / nonrecall / question positions -----
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

    q_recall = pick(a0 + recall_rel, a0 + recall_rel + 1)
    q_nonrecall = pick(a0 + nonrecall_rel, a0 + nonrecall_rel + 1)
    q_question = pick(0, len(pre))
    return [{"labels": labels, "k": kf, "gold": np.array(gold), "nB": np.array(nB),
             "q_question": q_question, "q_fire": q_recall,
             "q_fire_generic": q_recall, "q_ctrl": q_nonrecall,
             "fire_mass": np.array(float(blockmass[recall_rel].max()), dtype=np.float32),
             "ctrl_mass": np.array(float(blockmass[nonrecall_rel].max()), dtype=np.float32),
             "gold_fact": np.array(gold_fact),                # planted block (cross-check)
             "gold_is_fact": np.array(int(gold == gold_fact)),
             "to_evict_recall": np.array(float(to_evict[recall_rel]), dtype=np.float32),
             "seed": np.array(seed)}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--held", action="store_true")
    ap.add_argument("--out", default="cot_recall_labels_{shard}.npz")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(os.cpu_count())
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"COTSCAN_LABELS_V1 dev={dev} layers={LAYERS}", flush=True)
    tok = AutoTokenizer.from_pretrained("fft_hf")
    llm = AutoModelForCausalLM.from_pretrained(
        "fft_hf", dtype=torch.float32, attn_implementation="eager").eval().to(dev)
    pooler = load_pooler().to(dev)
    rows = []
    for i in range(args.n):
        seed = args.shard * 100000 + i
        try:
            rows += process(llm, tok, pooler, dev, seed, held=args.held)
        except Exception as e:
            print(f"seed {seed} failed: {e}", flush=True)
        if (i + 1) % 10 == 0:
            print(f"{i+1}/{args.n} -> {len(rows)} samples", flush=True)
    pack = {}
    for j, r in enumerate(rows):
        for k, v in r.items():
            pack[f"{j}_{k}"] = v
    pack["n_samples"] = np.array(len(rows))
    np.savez_compressed(args.out.format(shard=args.shard), **pack)
    print(f"COTSCAN_LABELS_DONE samples={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
