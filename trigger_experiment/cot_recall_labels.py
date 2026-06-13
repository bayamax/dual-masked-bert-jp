"""cot_recall_labels.py — COTRECALL_V1: recall a SP-compressed value MID-CoT.

The recall-query test (recall_query_labels.py) was biased: its question named the
fact ("what was the wifi password?"), so a question-position query had the cue handed
to it. The real target is different — an OBLIQUE compute question that needs a specific
value which now lives only in SP gist, recalled DURING the chain-of-thought:

    user (evicted): "By the way, the ferry ticket was 6800 yen round trip."
    ...filler scrolls it out of the window...
    user: "If I make that crossing four times, what will I pay in total?"   <- no value
    model CoT: "To total it I multiply the ferry ticket price by four, which is " | 6800
                                                                       fire position ^

Two things must hold for this to work, and this script measures BOTH on the same battery:

  (Q1 retrieval) Is the FIRE-position hidden state a better retrieval query than the
     diffuse question? Harvest q at question positions vs at the fire position (both under
     production [BOS][SP][window] context); train the Phase B indexer on each; top-2 gold.

  (Q2 mid-CoT gate) Can a per-position gate even FIRE here? Harvest full-KV attention mass
     to the evicted region per CoT position (the teacher signal) to show the recall need
     spikes at the fire position; and harvest SP-context features at fire vs a control CoT
     position so a linear probe AUC says whether inference can detect it.

Output: cot_recall_labels_<shard>.npz . Run: python3 cot_recall_labels.py --n 200 --shard 0
"""
import argparse, json, os, random, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

from attn_export3_torch import load_pooler
from phaseb_data import FILLER
from recall_query_labels import Capture

LAYERS = (8, 14, 20)
WINDOW, BLOCK = 512, 128

CQ = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "cot_questions.json")))
NEEDLES = CQ["needles"]            # [(statement_template, role), ...]
ITEMS = CQ["items"]               # [{"q":..., "lead":...}, ...]


def fill_needle(stmt, rng):
    """Fill the single numeric placeholder; return (filled_statement, value_string)."""
    val = None
    for ph, gen in (("{num4}", lambda: str(rng.randint(1000, 9999))),
                    ("{num3}", lambda: str(rng.randint(100, 999))),
                    ("{num2}", lambda: str(rng.randint(10, 99))),
                    ("{kg}", lambda: f"{rng.randint(2,60)}.{rng.randint(0,9)}")):
        if ph in stmt:
            val = gen()
            stmt = stmt.replace(ph, val)
    return stmt, val


def build_conversation(seed, item_idx, target=3800):
    """One needle planted early, padded with filler so it lands deep in the evicted
    region. Returns (pairs, needle_user, oblique_q, lead, value)."""
    rng = random.Random(seed)
    stmt, role = NEEDLES[item_idx]
    it = ITEMS[item_idx]
    filled, value = fill_needle(stmt, rng)
    if value is None:
        return None
    needle_user = f"By the way, {filled}."
    fillers = rng.sample(FILLER, len(FILLER))
    pairs, fi = [], 0
    pairs.append(fillers[fi % len(fillers)]); fi += 1
    pairs.append((needle_user, "Noted - thanks for telling me, that sounds handy."))
    est = sum(len((u + a).split()) for u, a in pairs) * 1.4
    lap = 0
    while est < target:
        u, a = fillers[fi % len(fillers)]
        if lap:
            u = "One more thing - " + u[0].lower() + u[1:]
        pairs.append((u, a)); fi += 1
        if fi % len(fillers) == 0:
            lap += 1
        est += len((u + a).split()) * 1.4
    return pairs, needle_user, it["q"], it["lead"], value


def to_ids(tok, pairs):
    ids, spans = [], []
    for u, a in pairs:
        text = ("<｜end▁of▁sentence｜>" if ids else "") + f"<｜User｜>{u}<｜Assistant｜>{a}"
        t = tok.encode(text, add_special_tokens=False)
        spans.append((len(ids), len(ids) + len(t)))
        ids.extend(t)
    return ids, spans


@torch.no_grad()
def generate_cot(llm, tok, dev, history_ids, pre, value, maxgen=90):
    """Full-KV greedy CoT: let the model reason (value IS visible in history) until it
    naturally emits `value`. Returns the generated ids BEFORE the value token (the
    natural fire prefix), or None if the model never reaches the value."""
    from transformers.cache_utils import DynamicCache
    cache = DynamicCache()
    out = llm(input_ids=torch.tensor([history_ids + pre], device=dev),
              past_key_values=cache, use_cache=True)
    nxt = int(out.logits[0, -1].argmax())
    gen, norm_val = [], value.replace(",", "")
    for _ in range(maxgen):
        gen.append(nxt)
        if norm_val in tok.decode(gen).replace(",", ""):
            # back off to the largest prefix that does NOT yet contain the value
            m = len(gen)
            while m > 0 and norm_val in tok.decode(gen[:m]).replace(",", ""):
                m -= 1
            return gen[:m]
        out = llm(input_ids=torch.tensor([[nxt]], device=dev),
                  past_key_values=cache, use_cache=True)
        nxt = int(out.logits[0, -1].argmax())
    return None


@torch.no_grad()
def process(llm, tok, pooler, dev, seed, natural=True):
    item_idx = seed % len(ITEMS)
    built = build_conversation(seed, item_idx)
    if built is None:
        return []
    pairs, needle_user, q_text, lead, value = built
    ids, spans = to_ids(tok, pairs)
    n_evict = ((len(ids) - WINDOW) // BLOCK) * BLOCK
    if n_evict < BLOCK * 4:
        return []
    nB = n_evict // BLOCK
    # gold block = block holding the needle utterance
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
    if natural:
        # let the real model write its OWN CoT (full-KV, value visible) and take the
        # point where it naturally reaches for the value as the fire position — removes
        # the hand-written-lead bias.
        lead_ids = generate_cot(llm, tok, dev, ids, pre, value)
        if not lead_ids:
            return []
    else:
        lead_ids = tok.encode(lead, add_special_tokens=False)
    val_ids = tok.encode(value, add_special_tokens=False)
    suffix = pre + lead_ids + val_ids
    q_start, fire_pos = 0, len(pre) + len(lead_ids) - 1     # last CoT token before value
    ctrl_pos = len(pre)                                      # first CoT token
    val_start = len(pre) + len(lead_ids)

    from transformers.cache_utils import DynamicCache
    # (1) full-KV: block keys + teacher attention (per-position to evicted) ------------
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
    fire_mass, ctrl_mass = [], []
    for lidx, li in enumerate(LAYERS):
        a = att[li][0].float()                              # [Hq, Tsuf, Tctx]
        # retrieval label: attention at the VALUE positions -> evicted blocks
        m = a[:, val_start:, :].sum(dim=(0, 1)); m[0] = 0.0
        ev = m[:n_evict].view(nB, BLOCK).sum(-1)
        if ev.sum().item() > 0:
            labels[lidx] = (ev / ev.sum()).cpu().numpy()
        # per-position mass to evicted (for the fire vs ctrl spike)
        def mass_to_evicted(pos):
            mm = a[:, pos, :].sum(0); mm[0] = 0.0
            return float(mm[:n_evict].sum() / max(mm.sum().item(), 1e-8))
        fire_mass.append(mass_to_evicted(fire_pos))
        ctrl_mass.append(mass_to_evicted(ctrl_pos))

    # (2) production context [BOS][SP][window]: q at question / fire / ctrl ------------
    kept, window = ids[:n_evict], ids[n_evict:]
    sp = pooler(embT(torch.tensor([kept], device=dev)).float()).to(embT.weight.dtype)
    win_emb = embT(torch.tensor([window], device=dev))
    cache = DynamicCache()
    llm.model(input_ids=torch.tensor([[bos]], device=dev), past_key_values=cache,
              use_cache=True)
    llm.model(inputs_embeds=torch.cat([sp, win_emb], 1), past_key_values=cache,
              use_cache=True)
    with Capture(llm, LAYERS, want_q=True, want_k=False) as cap2:
        llm.model(inputs_embeds=embT(torch.tensor([suffix], device=dev)),
                  past_key_values=cache, use_cache=True)
    qsp = cap2.q

    def pick(pos_a, pos_b=None):
        return np.stack([qsp[li][0][0][pos_a:pos_b].mean(0).view(Hq, D).cpu()
                         .float().numpy() for li in LAYERS]).astype(np.float16)

    q_question = pick(q_start, len(pre))                    # over the oblique question
    q_fire = pick(fire_pos, fire_pos + 1)
    q_ctrl = pick(ctrl_pos, ctrl_pos + 1)
    return [{"labels": labels, "k": kf, "gold": np.array(gold), "nB": np.array(nB),
             "q_question": q_question, "q_fire": q_fire, "q_ctrl": q_ctrl,
             "fire_mass": np.array(np.mean(fire_mass), dtype=np.float32),
             "ctrl_mass": np.array(np.mean(ctrl_mass), dtype=np.float32),
             "seed": np.array(seed), "item": np.array(item_idx)}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--out", default="cot_recall_labels_{shard}.npz")
    ap.add_argument("--template", action="store_true",
                    help="use the hand-written CoT lead instead of model-generated CoT")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(os.cpu_count())
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"COTRECALL_LABELS_V1 dev={dev} items={len(ITEMS)}", flush=True)
    tok = AutoTokenizer.from_pretrained("fft_hf")
    llm = AutoModelForCausalLM.from_pretrained(
        "fft_hf", dtype=torch.float32, attn_implementation="eager").eval().to(dev)
    pooler = load_pooler().to(dev)
    rows = []
    for i in range(args.n):
        seed = args.shard * 100000 + i
        try:
            rows += process(llm, tok, pooler, dev, seed, natural=not args.template)
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
    print(f"COTRECALL_LABELS_DONE samples={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
