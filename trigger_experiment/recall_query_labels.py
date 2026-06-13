"""recall_query_labels.py — RECALLQ_V1: which query vector retrieves the gold block best?

Same conversations / contextual block keys / teacher attention labels as Phase B
(phaseb_labels.py); the ONLY thing that varies is how the student QUERY is harvested.
Three variants per (conversation, question), so the indexer can be trained on each and
compared apples-to-apples:

  q_fullq : question-position pre-RoPE query under FULL-KV context   (Phase B baseline)
  q_spq   : question-position pre-RoPE query under PRODUCTION context [BOS][SP][window]
            (the V2 fix — what inference actually has at turn start)
  q_spans : ANSWER-position pre-RoPE query under PRODUCTION context  (the new idea:
            the hidden state mid-generation, "what I'm about to say", as the key)

Keys (k) and labels come from one full-context prefill, exactly as Phase B. Block keys
are contextual (mean+max pooled pre-RoPE k_proj over each evicted block).

Output: recall_query_labels_<shard>.npz . Run: python3 recall_query_labels.py --n 200 --shard 0
"""
import argparse, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

from attn_export3_torch import load_pooler
from phaseb_data import make_conversation, to_ids

LAYERS = (8, 14, 20)
WINDOW, BLOCK = 512, 128


class Capture:
    """Capture pre-RoPE q_proj and k_proj outputs of the selected layers."""
    def __init__(self, llm, layers, want_q=True, want_k=True):
        self.llm, self.layers = llm, layers
        self.want_q, self.want_k = want_q, want_k
        self.k, self.q, self.hooks = {}, {}, []

    def __enter__(self):
        for li in self.layers:
            attn = self.llm.model.layers[li].self_attn

            def mk(store, li):
                def fn(_m, _i, o):
                    store.setdefault(li, []).append(o.detach())
                return fn
            if self.want_k:
                self.hooks.append(attn.k_proj.register_forward_hook(mk(self.k, li)))
            if self.want_q:
                self.hooks.append(attn.q_proj.register_forward_hook(mk(self.q, li)))
        return self

    def __exit__(self, *a):
        for h in self.hooks:
            h.remove()


@torch.no_grad()
def process(llm, tok, pooler, dev, seed, heldout=False):
    pairs, questions = make_conversation(seed, heldout=heldout)
    ids, _ = to_ids(tok, pairs)
    n_evict = ((len(ids) - WINDOW) // BLOCK) * BLOCK
    if n_evict < BLOCK * 4:
        return []
    nB = n_evict // BLOCK
    cfg = llm.config
    Hq, Hkv = cfg.num_attention_heads, cfg.num_key_value_heads
    D = cfg.hidden_size // Hq
    embT = llm.get_input_embeddings()
    bos = tok.bos_token_id

    from transformers.cache_utils import DynamicCache
    # production-context prefix [BOS][SP(kept)][window] (shared across questions)
    kept, window = ids[:n_evict], ids[n_evict:]
    sp = pooler(embT(torch.tensor([kept], device=dev)).float()).to(embT.weight.dtype)
    win_emb = embT(torch.tensor([window], device=dev))

    out = []
    for q_text, needle_user in questions:
        gold, acc = None, []
        for u, a in pairs:
            text = ("<｜end▁of▁sentence｜>" if acc else "") + f"<｜User｜>{u}<｜Assistant｜>{a}"
            t = tok.encode(text, add_special_tokens=False)
            if u == needle_user:
                gold = (len(acc) + len(t) // 2) // BLOCK
            acc.extend(t)
        if gold is None or gold >= nB:
            continue
        pre = tok.encode(f"<｜end▁of▁sentence｜><｜User｜>{q_text}<｜Assistant｜>"
                         f"<think>\n\n</think>\n\n", add_special_tokens=False)
        ans = tok.encode(f"You told me earlier: {needle_user}", add_special_tokens=False)
        suffix, ans_start = pre + ans, len(pre)

        # (A) fresh full-KV prefill: contextual block keys + teacher labels + q_fullq.
        # Re-prefilled per question (Phase B does the same) — avoids cache copying and
        # is version-robust across transformers DynamicCache APIs.
        cache = DynamicCache()
        with Capture(llm, LAYERS, want_q=False, want_k=True) as cap:
            llm.model(input_ids=torch.tensor([ids], device=dev),
                      past_key_values=cache, use_cache=True)
        kfeat = {li: cap.k[li][0][0] for li in LAYERS}            # [T_hist, Hkv*D]
        kf = []
        for li in LAYERS:
            k = kfeat[li][:n_evict].view(nB, BLOCK, Hkv, D).float()
            kf.append(torch.cat([k.mean(1), k.amax(1)], dim=-1).cpu().numpy())
        kf = np.stack(kf, axis=1).astype(np.float16)             # [nB, L, Hkv, 2D]
        with Capture(llm, LAYERS, want_q=True, want_k=False) as cap2:
            o = llm.model(input_ids=torch.tensor([suffix], device=dev),
                          past_key_values=cache, use_cache=True, output_attentions=True)
        att = o.attentions
        qf_full = cap2.q                                          # [Tsuf, Hq*D]
        labels = np.zeros((len(LAYERS), nB), dtype=np.float32)
        for lidx, li in enumerate(LAYERS):
            a = att[li][0, :, ans_start:, :].float()
            mass = a.sum(dim=(0, 1)); mass[0] = 0.0
            ev = mass[:n_evict].view(nB, BLOCK).sum(-1)
            s = ev.sum().item()
            if s > 0:
                labels[lidx] = (ev / s).cpu().numpy()
        q_fullq = np.stack([qf_full[li][0][0][:ans_start].mean(0).view(Hq, D).cpu()
                            .float().numpy() for li in LAYERS])

        # (B) production-context pass: q_spq (question pos) + q_spans (answer pos)
        cache = DynamicCache()
        llm.model(input_ids=torch.tensor([[bos]], device=dev),
                  past_key_values=cache, use_cache=True)
        llm.model(inputs_embeds=torch.cat([sp, win_emb], 1),
                  past_key_values=cache, use_cache=True)
        with Capture(llm, LAYERS, want_q=True, want_k=False) as cap3:
            llm.model(inputs_embeds=embT(torch.tensor([suffix], device=dev)),
                      past_key_values=cache, use_cache=True)
        qf_sp = cap3.q
        q_spq = np.stack([qf_sp[li][0][0][:ans_start].mean(0).view(Hq, D).cpu()
                          .float().numpy() for li in LAYERS])
        q_spans = np.stack([qf_sp[li][0][0][ans_start:].mean(0).view(Hq, D).cpu()
                            .float().numpy() for li in LAYERS])

        out.append({"labels": labels, "k": kf,
                    "q_fullq": q_fullq.astype(np.float16),
                    "q_spq": q_spq.astype(np.float16),
                    "q_spans": q_spans.astype(np.float16),
                    "gold": np.array(gold), "nB": np.array(nB), "seed": np.array(seed)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--out", default="recall_query_labels_{shard}.npz")
    ap.add_argument("--heldout", action="store_true")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(os.cpu_count())
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"RECALLQ_LABELS_V1 dev={dev} layers={LAYERS}", flush=True)
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
    print(f"RECALLQ_LABELS_DONE samples={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
