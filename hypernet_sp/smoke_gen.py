"""Generation smoke A/B (CPU, torch): does the release decode policy (two-phase temperature
+ answer-oscillation budget forcing) help on the actual SP-evict pipeline?

Port of sp_mlx.generate_sp's bounded loop (B=1) to torch/fft_hf. Two arms per problem:
  base    constant temp 0.6 everywhere (shipping behaviour)
  policy  DecodePolicy: temp 0.6 inside <think>, ~greedy after; force-close </think> when
          the same asserted value appears k=3 times

Small n by necessity on CPU — this is a smoke test for wiring + direction, NOT a benchmark
(run evals/gsm8k_eval.py with the policy for real numbers on-device).

Run next to fft_hf/ and fft_out/pooler.pt:  python3 smoke_gen.py [--probs 3] [--cap 600]
"""
import argparse, os, sys, time
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attn_export3_torch import load_pooler
from decode_policy import DecodePolicy

BATTERY = [
    ("natalia", "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did she sell altogether in April and May?", ["72"]),
    ("train", "A train travels 60 km in 1.5 hours. At the same constant speed, how far does it travel in 4 hours?", ["160"]),
    ("discount", "After a 20% discount, a shirt costs $20. What was the original price?", ["25"]),
    ("mult", "What is 17 multiplied by 24?", ["408"]),
]


@torch.no_grad()
def generate(llm, tok, pooler, prompt, cap=600, rw=128, C=64, temp=0.6, policy=None, seed=0):
    torch.manual_seed(seed)
    embT = llm.get_input_embeddings()
    H = pooler.H
    eos = tok.eos_token_id
    think_close = tok.encode("\n</think>\n\nFinal answer: ", add_special_tokens=False)
    q_ids = tok.encode(f"<｜User｜>{prompt}<｜Assistant｜>", add_special_tokens=True) + \
        tok.encode("<think>\n", add_special_tokens=False)
    MQ = len(q_ids)
    cache = DynamicCache()
    llm(input_ids=torch.tensor([q_ids]), past_key_values=cache, use_cache=True)
    gen, kept, absorbed = [], [], 0
    in_think, forced, done = True, False, False
    feed = []
    t0 = time.time()
    while len(gen) < cap and not done:
        c0 = len(gen); R = min(c0, rw); nd_end = c0 - R
        if nd_end > absorbed:
            kept.extend(gen[absorbed:nd_end]); absorbed = nd_end
        sp = pooler(embT(torch.tensor([kept])).float()) if kept else \
            pooler(torch.zeros(1, 0, H))
        parts = [sp.to(embT.weight.dtype)]
        if R > 0:
            parts.append(embT(torch.tensor([gen[c0 - R:c0]])))
        block = torch.cat(parts, 1)
        cache.crop(MQ)
        last = llm(inputs_embeds=block, past_key_values=cache, use_cache=True).logits[:, -1, :].float()
        for _ in range(min(C, cap - len(gen))):
            if feed:
                t = feed.pop(0)
            else:
                T = policy.temp(in_think, temp) if policy else temp
                t = int(torch.multinomial(F.softmax(last[0] / T, -1), 1))
                if t == eos:
                    done = True; break
            gen.append(t)
            if in_think and "</think>" in tok.decode(gen[-8:]):
                in_think = False
            if policy and in_think and not feed:
                txt = tok.decode(gen)
                if policy.note_text(txt):
                    feed = list(think_close); forced = True; in_think = False
            last = llm(inputs_embeds=embT(torch.tensor([[t]])), past_key_values=cache,
                       use_cache=True).logits[:, -1, :].float()
        if done:
            break
    return {"text": tok.decode(gen), "n": len(gen), "eos": done, "forced": forced,
            "sec": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", type=int, default=3)
    ap.add_argument("--cap", type=int, default=600)
    a = ap.parse_args()
    torch.set_num_threads(os.cpu_count())
    tok = AutoTokenizer.from_pretrained("fft_hf")
    llm = AutoModelForCausalLM.from_pretrained("fft_hf", dtype=torch.float32).eval()
    pooler = load_pooler()
    score = {"base": 0, "policy": 0}
    toks = {"base": 0, "policy": 0}
    for name, q, want in BATTERY[:a.probs]:
        for arm in ("base", "policy"):
            pol = DecodePolicy() if arm == "policy" else None
            r = generate(llm, tok, pooler, q, cap=a.cap, policy=pol, seed=42)
            hit = all(w.lower() in r["text"].lower() for w in want)
            score[arm] += hit; toks[arm] += r["n"]
            tail = r["text"][-110:].replace("\n", " ")
            print(f"[{name}/{arm}] correct={hit} n={r['n']} eos={r['eos']} forced={r.get('forced')}"
                  f" ({r['sec']:.0f}s)\n   tail: {tail}", flush=True)
    n = a.probs
    print(f"\n== smoke result (n={n}, cap={a.cap}) ==")
    print(f"  base:   {score['base']}/{n} correct, {toks['base']} tokens total")
    print(f"  policy: {score['policy']}/{n} correct, {toks['policy']} tokens total")
    print("SMOKE_GEN_DONE")


if __name__ == "__main__":
    main()
