"""gsm8k_fullkv.py — the CONTROL: the SAME fft_hf model solving GSM8K with the FULL problem
in context (no SP compression, no eviction, no recall). This is the base-model ceiling.
If this is high while the SP+recall path is low, the cost is the pipeline, not the model.

Same model / greedy decoding / answer extraction as the recall runs, on the same 30 items.
"""
import os, re, json, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from transformers import DynamicCache
except Exception:
    from transformers.cache_utils import DynamicCache
from datasets import load_dataset

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CAP = int(os.environ.get("CAP", "512"))
N = int(os.environ.get("GSM_N", "30"))
OUT = os.environ.get("OUT_DIR", "results_gsm_fullkv")
LOG = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.append(s)
    os.makedirs(OUT, exist_ok=True); open(os.path.join(OUT, "live.log"), "w").write("\n".join(LOG))

INSTR = (" Reason step by step, then end with 'Final answer: ' followed by the single "
         "final number.")
GOLD_RE = re.compile(r"####\s*([-\d,\.]+)")
NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
def norm(s):
    s = s.replace(",", "").strip().rstrip(".")
    try:
        f = float(s); return str(int(f)) if f == int(f) else str(f)
    except Exception:
        return s
def final_num(t):
    m = NUM_RE.findall(t); return norm(m[-1]) if m else ""


@torch.no_grad()
def solve(tok, llm, problem):
    eos = tok.eos_token_id
    ids = tok.encode(f"<｜User｜>{problem}{INSTR}<｜Assistant｜><think>\n", add_special_tokens=True)
    cache = DynamicCache()
    last = llm(input_ids=torch.tensor([ids], device=DEV), past_key_values=cache,
               use_cache=True).logits[:, -1, :].float()
    gen = []
    for _ in range(CAP):
        t = int(last[0].argmax())
        if t == eos:
            break
        gen.append(t)
        last = llm(input_ids=torch.tensor([[t]], device=DEV), past_key_values=cache,
                   use_cache=True).logits[:, -1, :].float()
    body = tok.decode(gen)
    if "</think>" not in body:                      # close think + short greedy answer
        for t in tok.encode("\n</think>\n\nFinal answer: ", add_special_tokens=False):
            gen.append(t)
            last = llm(input_ids=torch.tensor([[t]], device=DEV), past_key_values=cache,
                       use_cache=True).logits[:, -1, :].float()
        for _ in range(24):
            t = int(last[0].argmax())
            if t == eos:
                break
            gen.append(t)
            last = llm(input_ids=torch.tensor([[t]], device=DEV), past_key_values=cache,
                       use_cache=True).logits[:, -1, :].float()
        body = tok.decode(gen)
    ans = body.split("</think>")[-1]
    if "Final answer:" in ans:
        ans = ans.split("Final answer:")[-1]
    return final_num(ans), len(gen)


def main():
    t0 = time.time(); os.makedirs(OUT, exist_ok=True)
    ds = load_dataset("gsm8k", "main", split="test")
    tok = AutoTokenizer.from_pretrained("fft_hf")
    try:
        llm = AutoModelForCausalLM.from_pretrained("fft_hf", torch_dtype=torch.float32)
    except TypeError:
        llm = AutoModelForCausalLM.from_pretrained("fft_hf", dtype=torch.float32)
    llm = llm.eval().to(DEV)
    log(f"[fullkv] N={N} cap={CAP}")
    correct = 0; results = []
    for i in range(N):
        g = GOLD_RE.search(ds[i]["answer"]); gold = norm(g.group(1)) if g else ""
        pred, ntok = solve(tok, llm, ds[i]["question"])
        ok = pred == gold and gold != ""
        correct += int(ok); results.append({"i": i, "gold": gold, "pred": pred, "correct": ok})
        log(f"[{i:2}] gold={gold:>6} pred={pred:>6} {'OK' if ok else 'x'}  [{i+1}/{N} t={time.time()-t0:.0f}s]")
        json.dump({"n": i + 1, "correct": correct, "acc": correct / (i + 1),
                   "results": results}, open(os.path.join(OUT, "results.json"), "w"), indent=1)
    log(f"\n=== FULL-KV GSM8K (no compression) ===\n  {correct}/{N} = {correct/N:.1%}")
    json.dump({"n": N, "correct": correct, "acc": correct / N, "results": results,
               "final": True}, open(os.path.join(OUT, "results.json"), "w"), indent=1)
    open(os.path.join(OUT, ".done"), "w").write("ok")
    log(f"DONE t={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
