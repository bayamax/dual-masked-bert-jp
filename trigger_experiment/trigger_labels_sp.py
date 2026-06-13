"""trigger_labels_sp.py — TRIGGER_V2: student features under the PRODUCTION context.

V1 measured teacher attention under full KV and harvested features under that same
full-KV cache (upper-bound form, like Phase B). Production inference never has that
cache: the question is encoded behind [BOS][SP (pooler over evicted)][raw window].
This script re-harvests the question-position pre-RoPE queries in exactly that
layout. No teacher pass is needed — the binary label is the question type, fixed by
construction and validated in V1 (oracle evicted-mass AUC = 1.0 on this battery).

Same seeds/shards as trigger_labels.py so rows correspond 1:1.
Output: trigger_labels_sp_<shard>.npz . Run: python3 trigger_labels_sp.py --n 200 --shard 0
"""
import argparse, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

from attn_export3_torch import load_pooler
from phaseb_data import to_ids
from trigger_labels import BLOCK, LAYERS, WINDOW, CaptureQ, make_trigger_conversation


@torch.no_grad()
def process(llm, tok, pooler, dev, seed, heldout=False):
    pairs, ev_questions, win_q, win_idx, generics = make_trigger_conversation(seed, heldout)
    ids, spans = to_ids(tok, pairs)
    n_evict = ((len(ids) - WINDOW) // BLOCK) * BLOCK
    if n_evict < BLOCK * 4:
        return []
    cfg = llm.config
    Hq = cfg.num_attention_heads
    D = cfg.hidden_size // Hq

    qlist = []
    needle_span = {u: spans[i] for i, (u, _a) in enumerate(pairs)}
    for q_text, needle_user in ev_questions:
        s, e = needle_span[needle_user]
        if e <= n_evict:
            qlist.append((q_text, 0))
    ws, _we = spans[win_idx]
    if ws >= n_evict:
        qlist.append((win_q[0], 1))
    qlist += [(q, 2) for q, _a in generics]

    # production layout (app_session_torch._gen_once): [BOS prime][SP(kept)][raw window]
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
        qfeat = {li: cap.q[li][0][0] for li in LAYERS}                # [Tpre, Hq*D]
        qf = np.stack([qfeat[li].mean(0).view(Hq, D).cpu().float().numpy()
                       for li in LAYERS])                             # [L, Hq, D]
        out.append({"fracs": np.zeros((len(LAYERS), 3), dtype=np.float32),
                    "q": qf.astype(np.float16),
                    "qtype": np.array(qtype), "seed": np.array(seed)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--out", default="trigger_labels_sp_{shard}.npz")
    ap.add_argument("--heldout", action="store_true")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(os.cpu_count())
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"TRIGGER_LABELS_SP_V2 dev={dev} layers={LAYERS}", flush=True)
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
