"""trigger_labels.py — teacher-label generation for the recall TRIGGER head (TRIGGER_V1).

Question: when the model answers, does its full-KV attention actually fly past the
exposure window into the evicted region? If yes, a windowed deployment would need a
block recall at that point. We measure this directly and train a head to predict it
from what inference actually has (question-position pre-RoPE queries, same student
feature as Phase B).

Three question types per conversation:
  0 EVICT   — recall question whose needle lies in the evicted region (positive)
  1 WINDOW  — recall question whose needle was planted inside the exposure window
              (hard negative: same surface form, answerable without recall)
  2 GENERIC — question with no past reference at all (easy negative)

For each question we teacher-force the gold answer with full context and record, per
layer, the answer-row attention mass split into (evicted, window, suffix) after BOS
sink exclusion. Student features are the question-position pre-RoPE queries (rows
before the answer start), mean-pooled — identical harvesting to phaseb_labels.py so
the head is comparable with the Phase B indexer pipeline.

Output: trigger_labels_<shard>.npz . Run:  python3 trigger_labels.py --n 200 --shard 0
"""
import argparse, os, random, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

from phaseb_data import (FILLER, HELDOUT_TEMPLATES, NEEDLE_TEMPLATES, _fill,
                         make_conversation, to_ids)

LAYERS = tuple(int(x) for x in os.environ.get("TRIGGER_LAYERS", "8,14,20").split(","))
WINDOW, BLOCK = 512, 128

GENERIC_QA = [
    ("Can you suggest a quick dinner idea for tonight?",
     "How about a simple vegetable stir fry over rice - quick, and you can use whatever protein you have on hand."),
    ("What is a good way to stay focused in the afternoon?",
     "Short breaks every half hour, a glass of water, and tackling one small task at a time usually helps."),
    ("Could you recommend a light exercise before bed?",
     "Gentle stretching or a short walk works well - it relaxes the muscles without raising your heart rate."),
    ("What snack goes well with green tea?",
     "Something lightly sweet like castella or rice crackers pairs nicely with green tea."),
    ("How should I store fresh basil so it lasts?",
     "Treat it like cut flowers - stems in a glass of water at room temperature, loosely covered with a bag."),
    ("What is a sensible default tip when unsure?",
     "Rounding up or about ten percent is a safe default in most casual settings where tipping is expected."),
    ("Any tips for falling asleep faster?",
     "Dim the lights an hour before bed, keep the room cool, and avoid screens - slow breathing also helps."),
    ("What makes a good cover letter opening?",
     "Lead with the specific role and one concrete thing you bring to it - skip generic introductions."),
]


def make_trigger_conversation(seed, heldout=False):
    """make_conversation + one extra needle planted near the end (inside the exposure
    window) for the hard-negative question. Replays make_conversation's template draw
    to avoid planting a template already used by an evicted needle."""
    pairs, questions = make_conversation(seed, heldout=heldout)
    pool = HELDOUT_TEMPLATES if heldout else NEEDLE_TEMPLATES
    used = random.Random(seed).sample(pool, min(3, len(pool)))  # same first draw
    avail = [t for t in pool if t not in used]
    rng = random.Random(seed * 7919 + 13)
    ut, qt = rng.choice(avail)
    joint = _fill(ut + "|||" + qt, rng)
    u, q = joint.split("|||")
    pairs.insert(len(pairs) - 1, (u, "Noted - thanks for telling me, that sounds like quite a day."))
    generics = rng.sample(GENERIC_QA, 2)
    return pairs, questions, (q, u), len(pairs) - 2, generics


class CaptureQ:
    def __init__(self, llm, layers):
        self.llm, self.layers, self.q, self.hooks = llm, layers, {}, []

    def __enter__(self):
        for li in self.layers:
            attn = self.llm.model.layers[li].self_attn

            def mk(store, li):
                def fn(_m, _i, o):
                    store.setdefault(li, []).append(o.detach())
                return fn
            self.hooks.append(attn.q_proj.register_forward_hook(mk(self.q, li)))
        return self

    def __exit__(self, *a):
        for h in self.hooks:
            h.remove()


@torch.no_grad()
def process(llm, tok, dev, seed, heldout=False):
    pairs, ev_questions, win_q, win_idx, generics = make_trigger_conversation(seed, heldout)
    ids, spans = to_ids(tok, pairs)
    n_evict = ((len(ids) - WINDOW) // BLOCK) * BLOCK
    if n_evict < BLOCK * 4:
        return []
    T_hist = len(ids)
    cfg = llm.config
    Hq = cfg.num_attention_heads
    D = cfg.hidden_size // Hq

    # build (question, gold answer, qtype), verifying needle placement by token span
    qlist = []
    needle_span = {u: spans[i] for i, (u, _a) in enumerate(pairs)}
    for q_text, needle_user in ev_questions:
        s, e = needle_span[needle_user]
        if e <= n_evict:                                       # actually evicted
            qlist.append((q_text, f"You told me earlier: {needle_user}", 0))
    ws, _we = spans[win_idx]
    if ws >= n_evict:                                          # actually in window
        qlist.append((win_q[0], f"You told me earlier: {win_q[1]}", 1))
    qlist += [(q, a, 2) for q, a in generics]

    from transformers.cache_utils import DynamicCache
    out = []
    for q_text, gold_answer, qtype in qlist:
        pre = tok.encode(f"<｜end▁of▁sentence｜><｜User｜>{q_text}<｜Assistant｜>"
                         f"<think>\n\n</think>\n\n", add_special_tokens=False)
        ans = tok.encode(gold_answer, add_special_tokens=False)
        suffix, ans_start_rel = pre + ans, len(pre)
        cache = DynamicCache()
        llm.model(input_ids=torch.tensor([ids], device=dev), past_key_values=cache,
                  use_cache=True)
        with CaptureQ(llm, LAYERS) as cap:
            o = llm.model(input_ids=torch.tensor([suffix], device=dev),
                          past_key_values=cache, use_cache=True, output_attentions=True)
        att = o.attentions                                     # L x [1,Hq,Tsuf,Tctx]
        qfeat = {li: cap.q[li][0][0] for li in LAYERS}         # [Tsuf, Hq*D]

        fracs = np.zeros((len(LAYERS), 3), dtype=np.float32)   # (evict, window, suffix)
        for lidx, li in enumerate(LAYERS):
            a = att[li][0, :, ans_start_rel:, :].float()       # [Hq, Tans, Tctx]
            mass = a.sum(dim=(0, 1))                           # [Tctx]
            mass[0] = 0.0                                      # BOS sink excluded
            ev = mass[:n_evict].sum().item()
            win = mass[n_evict:T_hist].sum().item()
            suf = mass[T_hist:].sum().item()
            tot = ev + win + suf
            if tot > 0:
                fracs[lidx] = [ev / tot, win / tot, suf / tot]
        qf = np.stack([qfeat[li][:ans_start_rel].mean(0).view(Hq, D).cpu().float().numpy()
                       for li in LAYERS])                      # [L, Hq, D]
        out.append({"fracs": fracs, "q": qf.astype(np.float16),
                    "qtype": np.array(qtype), "seed": np.array(seed)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--out", default="trigger_labels_{shard}.npz")
    ap.add_argument("--heldout", action="store_true")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(os.cpu_count())
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"TRIGGER_LABELS_V1 dev={dev} layers={LAYERS}", flush=True)
    tok = AutoTokenizer.from_pretrained("fft_hf")
    llm = AutoModelForCausalLM.from_pretrained(
        "fft_hf", dtype=torch.float32, attn_implementation="eager").eval().to(dev)
    rows = []
    for i in range(args.n):
        seed = args.shard * 100000 + i
        try:
            rows += process(llm, tok, dev, seed, heldout=args.heldout)
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
