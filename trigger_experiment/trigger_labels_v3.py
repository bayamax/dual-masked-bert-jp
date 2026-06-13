"""trigger_labels_v3.py — TRIGGER_V3: 4-class battery with expanded templates.

Adds the negative class that actually matters for the routing confusion the app
suffers from: LOOKUP (questions needing EXTERNAL/web knowledge — the trigger must
NOT fire on these; the intent router sends them to web). Also widens template
diversity with gpt-4o-mini-generated, mechanically validated templates
(battery_ext.json): needles 15+52, generic 8+25, filler 16+25, lookup 39.

Classes: 0 EVICT, 1 WINDOW, 2 GENERIC, 3 LOOKUP.
Features: production layout [BOS][SP(kept)][raw window][question] (V2 form).
Labels are the class by construction (V1 validated the premise: oracle AUC 1.0).

Template splits (train shard / --heldout shard):
  needles: orig 15 + ext[:35]  /  orig heldout 5 + ext[35:]
  generic: orig 8 + ext[:15]   /  ext[15:]
  lookup:  ext[:27]            /  ext[27:]
  filler:  shared (not label-bearing)

Output: trigger_labels_v3_<shard>.npz . Run: python3 trigger_labels_v3.py --n 200 --shard 0
"""
import argparse, json, os, random, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

import phaseb_data
import trigger_labels as TL
from attn_export3_torch import load_pooler
from phaseb_data import to_ids
from trigger_labels import BLOCK, LAYERS, WINDOW, CaptureQ

EXT = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "battery_ext.json")))
_NEEDLE_EXT = [(it["u"], it["q"]) for it in EXT["needles"]]
_ORIG_NEEDLE = list(phaseb_data.NEEDLE_TEMPLATES)
_ORIG_HELDOUT = list(phaseb_data.HELDOUT_TEMPLATES)
TRAIN_NEEDLES = _ORIG_NEEDLE + _NEEDLE_EXT[:35]
HELD_NEEDLES = _ORIG_HELDOUT + _NEEDLE_EXT[35:]
FILLER_ALL = list(phaseb_data.FILLER) + [(it["u"], it["a"]) for it in EXT["filler"]]
TRAIN_GENERIC = TL.GENERIC_QA + [(it["q"], it["a"]) for it in EXT["generic"][:15]]
HELD_GENERIC = [(it["q"], it["a"]) for it in EXT["generic"][15:]]
TRAIN_LOOKUP = [(it["q"], it["a"]) for it in EXT["lookup"][:27]]
HELD_LOOKUP = [(it["q"], it["a"]) for it in EXT["lookup"][27:]]


def _patch_pools(heldout):
    """make_conversation / make_trigger_conversation read module globals at call
    time — point both modules at the expanded pools."""
    phaseb_data.FILLER = FILLER_ALL
    if heldout:
        phaseb_data.NEEDLE_TEMPLATES = TRAIN_NEEDLES   # unused in heldout path
        phaseb_data.HELDOUT_TEMPLATES = HELD_NEEDLES
    else:
        phaseb_data.NEEDLE_TEMPLATES = TRAIN_NEEDLES
        phaseb_data.HELDOUT_TEMPLATES = _ORIG_HELDOUT
    TL.NEEDLE_TEMPLATES = phaseb_data.NEEDLE_TEMPLATES
    TL.HELDOUT_TEMPLATES = phaseb_data.HELDOUT_TEMPLATES
    TL.FILLER = phaseb_data.FILLER


@torch.no_grad()
def process(llm, tok, pooler, dev, seed, heldout=False):
    _patch_pools(heldout)
    pairs, ev_questions, win_q, win_idx, generics = \
        TL.make_trigger_conversation(seed, heldout)
    ids, spans = to_ids(tok, pairs)
    n_evict = ((len(ids) - WINDOW) // BLOCK) * BLOCK
    if n_evict < BLOCK * 4:
        return []
    cfg = llm.config
    Hq = cfg.num_attention_heads
    D = cfg.hidden_size // Hq

    rng = random.Random(seed * 31337 + 7)
    gen_pool = HELD_GENERIC if heldout else TRAIN_GENERIC
    look_pool = HELD_LOOKUP if heldout else TRAIN_LOOKUP
    qlist = []
    needle_span = {u: spans[i] for i, (u, _a) in enumerate(pairs)}
    for q_text, needle_user in ev_questions:
        s, e = needle_span[needle_user]
        if e <= n_evict:
            qlist.append((q_text, 0))
    ws, _we = spans[win_idx]
    if ws >= n_evict:
        qlist.append((win_q[0], 1))
    qlist += [(q, 2) for q, _a in rng.sample(gen_pool, 2)]
    qlist += [(q, 3) for q, _a in rng.sample(look_pool, 2)]

    embT = llm.get_input_embeddings()
    kept, window = ids[:n_evict], ids[n_evict:]
    sp = pooler(embT(torch.tensor([kept], device=dev)).float()).to(embT.weight.dtype)
    win_emb = embT(torch.tensor([window], device=dev))
    bos = tok.bos_token_id

    from transformers.cache_utils import DynamicCache
    out = []
    for q_text, qtype in qlist:
        pre = tok.encode(f"<｜end▁of▁sentence｜><｜User｜>{q_text}<｜Assistant｜>"
                         f"<think>\n\n</think>\n\n", add_special_tokens=False)
        cache = DynamicCache()
        llm.model(input_ids=torch.tensor([[bos]], device=dev), past_key_values=cache,
                  use_cache=True)
        llm.model(inputs_embeds=torch.cat([sp, win_emb], 1), past_key_values=cache,
                  use_cache=True)
        with CaptureQ(llm, LAYERS) as cap:
            llm.model(inputs_embeds=embT(torch.tensor([pre], device=dev)),
                      past_key_values=cache, use_cache=True)
        qfeat = {li: cap.q[li][0][0] for li in LAYERS}
        qf = np.stack([qfeat[li].mean(0).view(Hq, D).cpu().float().numpy()
                       for li in LAYERS])
        out.append({"fracs": np.zeros((len(LAYERS), 3), dtype=np.float32),
                    "q": qf.astype(np.float16),
                    "qtype": np.array(qtype), "seed": np.array(seed)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--out", default="trigger_labels_v3_{shard}.npz")
    ap.add_argument("--heldout", action="store_true")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(os.cpu_count())
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"TRIGGER_LABELS_V3 dev={dev} layers={LAYERS} "
          f"pools: needles {len(TRAIN_NEEDLES)}/{len(HELD_NEEDLES)} "
          f"generic {len(TRAIN_GENERIC)}/{len(HELD_GENERIC)} "
          f"lookup {len(TRAIN_LOOKUP)}/{len(HELD_LOOKUP)} filler {len(FILLER_ALL)}",
          flush=True)
    tok = AutoTokenizer.from_pretrained("fft_hf")
    llm = AutoModelForCausalLM.from_pretrained(
        "fft_hf", dtype=torch.float32, attn_implementation="eager").eval().to(dev)
    pooler = load_pooler().to(dev)
    rows = []
    for i in range(args.n):
        seed = args.shard * 100000 + i
        try:
            rows += process(llm, tok, pooler, dev, seed, heldout=args.heldout)
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
    print(f"TRIGGER_LABELS_DONE samples={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
