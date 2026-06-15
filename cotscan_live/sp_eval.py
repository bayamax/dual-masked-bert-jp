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
RW = int(os.environ.get("RW", "200"))         # raw window (production 512; smaller forces eviction on CPU)
PHASE_A = int(os.environ.get("PHASE_A", "200"))  # CoT tokens generated WITH the problem in context
REMAIN = int(os.environ.get("REMAIN", "240"))    # CoT tokens generated AFTER eviction
GSM_N = int(os.environ.get("GSM_N", "5"))
ARMS = os.environ.get("ARMS", "OFF,PIN").split(",")
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
def _gen(llm, ids, n, eos):
    out = llm.generate(input_ids=torch.tensor([ids], device=DEV), max_new_tokens=n,
                       do_sample=False, pad_token_id=eos)
    return out[0, len(ids):].tolist()


@torch.no_grad()
def phase_a(tok, llm, problem_ids):
    """Generate the first PHASE_A CoT tokens WITH the problem in context (one call)."""
    return _gen(llm, problem_ids, PHASE_A, tok.eos_token_id)


@torch.no_grad()
def phase_b(tok, llm, problem_ids, a_gen, arm):
    """Evict: keep only the last RW tokens of (problem + phaseA CoT). The problem (at the start)
    drops out when total > RW. Then continue REMAIN tokens. OFF = no recall; PIN = re-inject the
    evicted problem block once and keep it pinned. Returns (full_text, evicted?)."""
    eos = tok.eos_token_id
    full = problem_ids + a_gen
    evicted = len(full) > RW
    window = full[-RW:]
    visible = (problem_ids + window) if (arm == "PIN" and evicted) else window
    b_gen = _gen(llm, visible, REMAIN, eos)
    return tok.decode(a_gen + b_gen, skip_special_tokens=True), evicted


def main():
    t0 = time.time()
    tok, llm = build()
    log(f"[sp] loaded {time.time()-t0:.0f}s | RW={RW} PHASE_A={PHASE_A} REMAIN={REMAIN} arms={ARMS}")
    gsm = load_dataset("gsm8k", "main", split=f"test[:{GSM_N}]")
    items = [(ex["question"], ex["answer"].split("####")[-1].strip().replace(",", "")) for ex in gsm]

    res = {a: {"correct": 0, "evicted": 0, "rows": []} for a in ARMS}
    for qi, (q, gold) in enumerate(items):
        pid = tok.encode(f"<｜begin▁of▁sentence｜><｜User｜>{q}<｜Assistant｜>", add_special_tokens=False)
        a_gen = phase_a(tok, llm, pid)                      # shared CoT prefix across arms
        for a in ARMS:
            txt, evic = phase_b(tok, llm, pid, a_gen, a)
            pred = last_int(txt); ok = (pred == gold)
            res[a]["correct"] += int(ok); res[a]["evicted"] += int(evic)
            res[a]["rows"].append({"gold": gold, "pred": pred, "ok": ok, "evicted": evic})
            log(f"  q{qi} [{a:4}] evicted={int(evic)} pred={pred} gold={gold} {'OK' if ok else ''}")

    summary = {a: {"acc": round(res[a]["correct"] / len(items), 3),
                   "n": len(items), "evicted": res[a]["evicted"]} for a in ARMS}
    rep = {"model": MODEL, "RW": RW, "PHASE_A": PHASE_A, "REMAIN": REMAIN, "gsm_n": len(items),
           "note": "CPU 2-phase window-eviction proxy for SP; oracle recall (re-inject problem post-eviction)",
           "summary": summary, "detail": res, "elapsed_s": round(time.time() - t0, 1)}
    json.dump(rep, open(os.path.join(OUT, "results.json"), "w"), indent=1)
    log("\n=== SP-REGIME GSM8K (recall necessity under eviction) ===")
    for a in ARMS:
        log(f"  {a:4}: acc={summary[a]['acc']}  (evicted {summary[a]['evicted']}/{summary[a]['n']})")
    log("  expect: OFF low (problem evicted) | PIN restores (recall mandatory under SP)")
    log(f"DONE t={rep['elapsed_s']}s")


if __name__ == "__main__":
    main()
