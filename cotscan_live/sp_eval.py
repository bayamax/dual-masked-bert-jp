"""sp_eval.py — evaluate the composite UNDER THE SP/EVICTION REGIME (the regime that matters).

Correction to the earlier (wrong) framing: while doing CoT, the raw window fills with reasoning
and the ORIGINAL PROBLEM (operands) is pushed out -> evicted -> in production SP-compressed, which
loses high-entropy specifics. So for CoT tasks (GSM8K, Dolphin) RECALL IS MANDATORY, not a
false-fire to suppress. cotscan showed OFF (no recall) = 0/25 on GSM8K; recall restores it.

CPU proxy (no private SP pooler): a chunked generation loop with a raw window of RW tokens. Once
problem+CoT exceeds RW the problem is evicted (dropped — strictly harsher than SP summary). Arms:
  OFF  : no recall (problem stays evicted) -> should fail
  SWAP : re-inject the problem block EVERY chunk (naive per-position) -> cotscan found unstable
  PIN  : inject once when first evicted, keep it pinned -> cotscan's stable recipe
Recall timing uses an ORACLE (fire when the problem is evicted) to isolate the INJECTION strategy
from gate precision (web/recall detection separability already validated separately). GSM8K acc
per arm shows recall necessity + which injection strategy is sane. This sets up web coexistence.
"""
import os, json, time, re
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL = os.environ.get("MODEL", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
RW = int(os.environ.get("RW", "320"))       # raw window (production uses 512; smaller forces eviction on CPU)
CAP = int(os.environ.get("CAP", "448"))     # max new tokens
CHUNK = int(os.environ.get("CHUNK", "32"))  # rebuild window every CHUNK tokens
GSM_N = int(os.environ.get("GSM_N", "5"))
ARMS = os.environ.get("ARMS", "OFF,SWAP,PIN").split(",")
OUT = os.environ.get("OUT_DIR", "results_sp")
DEV = "cpu"
os.makedirs(OUT, exist_ok=True)
LOG = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.append(s)
    open(os.path.join(OUT, "live.log"), "w").write("\n".join(LOG))


def build():
    tok = AutoTokenizer.from_pretrained(MODEL)
    try:
        llm = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32, attn_implementation="eager")
    except TypeError:
        llm = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, attn_implementation="eager")
    return tok, llm.eval().to(DEV)


def last_int(t):
    nums = re.findall(r"-?\d[\d,]*", t.replace("$", ""))
    return nums[-1].replace(",", "") if nums else None


@torch.no_grad()
def gen_evict(tok, llm, problem_ids, arm):
    """Chunked generation with a raw window of RW. problem_ids sits at the START so it evicts first.
    Returns (text, n_recall_injects, ever_evicted)."""
    eos = tok.eos_token_id
    gen, pinned, n_inj, ever_eviu = [], None, 0, False
    while len(gen) < CAP:
        full = problem_ids + gen
        evicted = len(full) > RW
        ever_eviu = ever_eviu or evicted
        window = full[-RW:]
        inject = []
        if evicted and arm in ("SWAP", "PIN"):
            if arm == "SWAP":
                inject = problem_ids; n_inj += 1
            else:  # PIN: capture once, reuse
                if pinned is None:
                    pinned = list(problem_ids)
                inject = pinned; n_inj += 1
        visible = inject + window
        ids = torch.tensor([visible], device=DEV)
        out = llm.generate(input_ids=ids, max_new_tokens=CHUNK, do_sample=False, pad_token_id=eos)
        new = out[0, len(visible):].tolist()
        # stop at eos
        if eos in new:
            new = new[:new.index(eos)]; gen += new; break
        gen += new
        if not new:
            break
    return tok.decode(gen, skip_special_tokens=True), n_inj, ever_eviu


def main():
    t0 = time.time()
    tok, llm = build()
    log(f"[sp] loaded {time.time()-t0:.0f}s | RW={RW} CAP={CAP} CHUNK={CHUNK} arms={ARMS}")
    gsm = load_dataset("gsm8k", "main", split=f"test[:{GSM_N}]")
    items = [(ex["question"], ex["answer"].split("####")[-1].strip().replace(",", "")) for ex in gsm]

    res = {a: {"correct": 0, "evicted": 0, "rows": []} for a in ARMS}
    for qi, (q, gold) in enumerate(items):
        pid = tok.encode(f"<｜begin▁of▁sentence｜><｜User｜>{q}<｜Assistant｜>", add_special_tokens=False)
        for a in ARMS:
            txt, ninj, evic = gen_evict(tok, llm, pid, a)
            pred = last_int(txt); ok = (pred == gold)
            res[a]["correct"] += int(ok); res[a]["evicted"] += int(evic)
            res[a]["rows"].append({"gold": gold, "pred": pred, "ok": ok, "injects": ninj, "evicted": evic})
            log(f"  q{qi} [{a:4}] evicted={int(evic)} injects={ninj:>2} pred={pred} gold={gold} {'OK' if ok else ''}")

    summary = {a: {"acc": round(res[a]["correct"] / len(items), 3),
                   "n": len(items), "evicted": res[a]["evicted"]} for a in ARMS}
    rep = {"model": MODEL, "RW": RW, "CAP": CAP, "CHUNK": CHUNK, "gsm_n": len(items),
           "note": "CPU window-eviction proxy for SP; recall timing = oracle (fire when problem evicted)",
           "summary": summary, "detail": res, "elapsed_s": round(time.time() - t0, 1)}
    json.dump(rep, open(os.path.join(OUT, "results.json"), "w"), indent=1)
    log("\n=== SP-REGIME GSM8K (recall necessity + injection strategy) ===")
    for a in ARMS:
        log(f"  {a:4}: acc={summary[a]['acc']}  (evicted {summary[a]['evicted']}/{summary[a]['n']})")
    log("  expect: OFF low (problem evicted) | SWAP unstable | PIN best")
    log(f"DONE t={rep['elapsed_s']}s")


if __name__ == "__main__":
    main()
