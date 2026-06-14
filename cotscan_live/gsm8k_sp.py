"""gsm8k_sp.py — solve GSM8K NORMALLY through the SP generation runtime (no filler, no
recall). The problem is the user turn; the model reasons. As the CoT grows past rw, the
early reasoning is absorbed into the SP gist (exactly _gen_once). If the CoT fits within
rw, nothing is compressed and it's ~full-KV. Tests: does SP@rw degrade GSM8K vs full-KV?

Same fft_hf / greedy / scoring as gsm8k_fullkv.py. Set RW via env (512, 1024, ...).
"""
import os, re, json, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from transformers import DynamicCache
except Exception:
    from transformers.cache_utils import DynamicCache
from datasets import load_dataset
from attn_export3_torch import load_pooler

DEV = "cuda" if torch.cuda.is_available() else "cpu"
RW = int(os.environ.get("RW", "1024"))
C, CAP = 64, int(os.environ.get("CAP", "512"))
N = int(os.environ.get("GSM_N", "30"))
OUT = os.environ.get("OUT_DIR", "results_gsm_sp")
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


def emb(embT, ids):
    if ids:
        return embT(torch.tensor([ids], device=DEV))
    return torch.zeros(1, 0, embT.weight.shape[1], dtype=embT.weight.dtype, device=DEV)


@torch.no_grad()
def solve(tok, llm, pooler, embT, problem):
    eos = tok.eos_token_id
    THINK = tok.encode("<think>\n", add_special_tokens=False)
    feed = tok.encode(f"<｜User｜>{problem}{INSTR}<｜Assistant｜>", add_special_tokens=False) + list(THINK)
    gen, kept, absorbed = [], [], 0
    fi = new = 0; done = False; max_kept = 0
    cache = DynamicCache()
    bos = tok.bos_token_id if tok.bos_token_id is not None else eos
    prime_last = llm(input_ids=torch.tensor([[bos]], device=DEV),
                     past_key_values=cache, use_cache=True).logits[:, -1, :].float()
    MQ = cache.get_seq_length()
    while not done:
        c0 = len(gen); R = min(c0, RW); nd = c0 - R
        if nd > absorbed:
            kept.extend(gen[absorbed:nd]); absorbed = nd; max_kept = max(max_kept, len(kept))
        parts = []
        if kept:                                    # SP gist of the scrolled-out reasoning
            parts.append(pooler(emb(embT, kept).float()).to(embT.weight.dtype))
        if R > 0:
            parts.append(emb(embT, gen[c0 - R:c0]))
        cache.crop(MQ)
        last = (llm(inputs_embeds=torch.cat(parts, 1), past_key_values=cache,
                    use_cache=True).logits[:, -1, :].float() if parts else prime_last)
        for _ in range(C):
            if fi < len(feed):
                t = feed[fi]; fi += 1
            else:
                t = int(last[0].argmax())
                if t == eos:
                    done = True; break
                new += 1
                if new >= CAP:
                    done = True; break
            gen.append(t)
            last = llm(inputs_embeds=emb(embT, [t]), past_key_values=cache,
                       use_cache=True).logits[:, -1, :].float()
        if done:
            break
    body = tok.decode(gen)
    if "</think>" not in body:
        for t in tok.encode("\n</think>\n\nFinal answer: ", add_special_tokens=False):
            gen.append(t); last = llm(inputs_embeds=emb(embT, [t]), past_key_values=cache,
                                      use_cache=True).logits[:, -1, :].float()
        for _ in range(24):
            t = int(last[0].argmax())
            if t == eos:
                break
            gen.append(t); last = llm(inputs_embeds=emb(embT, [t]), past_key_values=cache,
                                      use_cache=True).logits[:, -1, :].float()
        body = tok.decode(gen)
    ans = body.split("</think>")[-1]
    if "Final answer:" in ans:
        ans = ans.split("Final answer:")[-1]
    return final_num(ans), len(gen), max_kept


def main():
    t0 = time.time(); os.makedirs(OUT, exist_ok=True)
    ds = load_dataset("gsm8k", "main", split="test")
    tok = AutoTokenizer.from_pretrained("fft_hf")
    try:
        llm = AutoModelForCausalLM.from_pretrained("fft_hf", torch_dtype=torch.float32)
    except TypeError:
        llm = AutoModelForCausalLM.from_pretrained("fft_hf", dtype=torch.float32)
    llm = llm.eval().to(DEV)
    pooler = load_pooler().to(DEV).eval()
    embT = llm.get_input_embeddings()
    log(f"[sp] RW={RW} N={N} cap={CAP}")
    correct = 0; compressed = 0; results = []
    for i in range(N):
        g = GOLD_RE.search(ds[i]["answer"]); gold = norm(g.group(1)) if g else ""
        try:
            pred, ntok, mk = solve(tok, llm, pooler, embT, ds[i]["question"])
        except Exception as e:
            log(f"  item {i} failed: {e}"); pred, ntok, mk = "", 0, 0
        ok = pred == gold and gold != ""
        correct += int(ok); compressed += int(mk > 0)
        results.append({"i": i, "gold": gold, "pred": pred, "correct": ok, "tokens": ntok, "kept": mk})
        log(f"[{i:2}] gold={gold:>6} pred={pred:>6} {'OK' if ok else 'x'} tok={ntok} kept={mk}  [{i+1}/{N} t={time.time()-t0:.0f}s]")
        json.dump({"n": i+1, "correct": correct, "acc": correct/(i+1), "rw": RW,
                   "compressed_count": compressed, "results": results},
                  open(os.path.join(OUT, "results.json"), "w"), indent=1)
    log(f"\n=== SP@rw={RW} GSM8K ===\n  {correct}/{N} = {correct/N:.1%}  ({compressed}/{N} had CoT>rw -> SP-compressed)")
    json.dump({"n": N, "correct": correct, "acc": correct/N, "rw": RW,
               "compressed_count": compressed, "results": results, "final": True},
              open(os.path.join(OUT, "results.json"), "w"), indent=1)
    open(os.path.join(OUT, ".done"), "w").write("ok")
    log(f"DONE t={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
