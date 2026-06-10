"""Composite battery v5 — LONG-HAUL: drive the generation stream to 6k, then ~12k tokens
(long-CoT multi-part problems x multi-turn) and measure what no earlier battery touched:

  * mass-based EVICTION (maxD=4096): fires when absorbed-distant exceeds the bound — the
    production path RESULTS.md flags as 'only activates at extreme buffer sizes'
  * quality checkpoints at ~6k and ~12k: anchor-fact recall, math on a corrected value,
    a gist probe ('what topics have we covered?') that leans on the SP, honest miss
  * termination health of every volume turn (think must close; no runaway)
  * tok/s flatness across the haul (bounded KV + bounded pooler input)

Run next to fft_hf/:  python3 composite_test5.py [--target 12000] [--cap 1600]
"""
import argparse, json, os, re, sys, time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "runtime"))
import memory_core as mc
from app_session_torch import AppSession
from decode_policy import DecodePolicy

results, vol_stats, ANSWERS = [], [], {}

# multi-part problems: long natural CoT, several DISTINCT sub-answers (so the convergence
# trigger doesn't cut the turn short), one checkable final value each.
VOLUME = [
 ("P1", "I invest $2,000 at 5% compound interest per year. (a) What is it worth after 1 year? (b) After 2 years? (c) After 3 years? Work each part out carefully.", ["2315.25"]),
 ("P2", "A train goes 60 km at 30 km/h, then 90 km at 45 km/h, then rests 30 minutes, then 100 km at 50 km/h. (a) Time for each leg? (b) Total trip time including the rest? (c) Average speed counting the rest?", ["6.5", "38"]),
 ("P3", "A committee of 3 is chosen from 5 women and 4 men. (a) How many total committees? (b) How many with exactly 2 women? (c) How many with at least 1 man?", ["84", "40", "74"]),
 ("P4", "A $1,200 paycheck is split: 30% rent, 25% food, 15% savings, the rest entertainment. (a) Dollar amount of each category? (b) If rent rises to 35%, how much LESS goes to entertainment?", ["360", "60"]),
 ("P5", "A rectangle is 12 cm by 8 cm. (a) Its area and perimeter? (b) A circle of radius 4 cm is cut from it - remaining area, using pi=3.14? (c) What percent of the rectangle was removed, to one decimal?", ["45.7", "52.3"]),
 ("P6", "Find the smallest positive number that leaves remainder 2 when divided by 3, remainder 3 when divided by 5, and remainder 2 when divided by 7. Verify each condition.", ["23"]),
 ("P7", "Pipe A fills a tank in 6 hours, pipe B in 4 hours, and drain C empties it in 12 hours. (a) Rate of each per hour as a fraction? (b) All three open: how long to fill? (c) A and B only: how long?", ["2.4"]),
 ("P8", "A bag has 5 red, 4 blue, 3 green marbles. Two are drawn without replacement. (a) P(both red)? (b) P(one red one blue)? (c) P(no green)? Give each as a fraction and verify.", ["5/33", "10/33"]),
]


def run(sess, msg, store, name, want=None, custom=None, volume=False, cap=None):
    t0 = time.time()
    n0 = len(sess.gen)
    if volume:
        ans, src, chunks = sess.turn(msg, store=store)
    else:
        ans, src, chunks = sess.turn(msg, store=store)
    dt, ntok = time.time() - t0, len(sess.gen) - n0
    checks = {}
    if want is not None:
        checks["answer"] = any(w.lower() in ans.lower() for w in want)
    if custom is not None:
        checks["custom"] = custom(ans)
    ok = all(checks.values()) if checks else True
    results.append((name, ok, checks))
    ANSWERS[name] = ans
    if volume:
        body = sess.tok.decode(sess.gen[n0:])
        closed = "</think>" in body
        vol_stats.append((name, ntok, dt, ntok / max(dt, 1), closed))
        results.append((f"{name}.terminated", closed, {}))
    print(f"[{name}] {'PASS' if ok else 'FAIL'} {checks} ({dt:.0f}s, +{ntok} tok, "
          f"stream={len(sess.gen)}, kept={len(sess.kept)}, evictions={sess.evictions})\n"
          f"   ans={ans[:120]!r}", flush=True)
    return ans


def probes(s, tag):
    run(s, "What's my reservation code?", "none", f"{tag}.code", want=["QX7-2291"])
    run(s, "If I spend $200 from my budget, how much is left?", "none", f"{tag}.math-budget",
        want=["450"])
    run(s, "In one sentence each, what topics have we worked through so far?", "none",
        f"{tag}.gist", custom=lambda a: sum(k in a.lower() for k in
            ("invest", "interest", "train", "committee", "paycheck", "rent", "rectangle",
             "circle", "remainder", "pipe", "tank", "marble", "probability", "budget")) >= 2)
    run(s, "What's my blood type?", "none", f"{tag}.honest-miss",
        custom=lambda a: "don't have that saved" in a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=12000)
    ap.add_argument("--cap", type=int, default=1600)
    a = ap.parse_args()
    torch.set_num_threads(os.cpu_count())
    import app_session_torch
    app_session_torch.ANSCAP = 4000   # the app's 600-char UX cap truncates part (c) of a
    # multi-part answer BEFORE the gold check sees it (P1: the 3-year value fell past the
    # cap). Benchmarks raise it (cf. gsm8k_eval raising T.ANSCAP); so does this battery.
    # App-side recommendation: make ANSCAP adaptive for multi-part questions.
    import joblib
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from rag import BGERetriever
    sys.path.pop(1)
    from attn_export3_torch import load_pooler

    tok = AutoTokenizer.from_pretrained("fft_hf")
    llm = AutoModelForCausalLM.from_pretrained("fft_hf", dtype=torch.float32).eval()
    pooler, bge = load_pooler(), BGERetriever()
    mem = mc.TieredMemory("/dev/null", bge=bge)
    s = AppSession(llm, tok, pooler, bge, joblib.load("evals/intent_clf.joblib"),
                   joblib.load("evals/specificity_clf.joblib"), mem, cap=a.cap, seed=41)

    print("#### anchors ####", flush=True)
    for f in ["My reservation code is QX7-2291.", "I'm allergic to peanuts.",
              "My gift budget is $650.", "The project is called Apollo."]:
        run(s, f, "session", "anchor")

    half = a.target // 2
    vi = 0
    print(f"\n#### volume to ~{half} tok ####", flush=True)
    while len(s.gen) < half and vi < len(VOLUME):
        name, q, want = VOLUME[vi]; vi += 1
        run(s, q, "none", name, want=want, volume=True)
    print(f"\n#### checkpoint A @ {len(s.gen)} tok ####", flush=True)
    probes(s, "A")

    print(f"\n#### volume to ~{a.target} tok ####", flush=True)
    while len(s.gen) < a.target and vi < len(VOLUME):
        name, q, want = VOLUME[vi]; vi += 1
        run(s, q, "none", name, want=want, volume=True)
    print(f"\n#### checkpoint B @ {len(s.gen)} tok ####", flush=True)
    probes(s, "B")
    run(s, "Am I allergic to anything?", "none", "B.allergy", want=["peanut"])

    print("\n" + "=" * 70, flush=True)
    for name, ok, _ in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print("\nvolume turns (tok, sec, tok/s, think-closed):")
    for name, ntok, dt, rate, closed in vol_stats:
        print(f"  {name}: {ntok:>5} tok  {dt:>5.0f}s  {rate:.2f} tok/s  closed={closed}")
    print(f"\nfinal: stream={len(s.gen)} tok | kept={len(s.kept)} | absorbed={s.absorbed} "
          f"| evictions={s.evictions} | maxD={s.maxD}")
    npass = sum(1 for _, ok, _ in results if ok)
    print(f"\nCOMPOSITE5: {npass}/{len(results)} PASS")
    json.dump({"results": [{"name": n, "ok": o} for n, o, _ in results],
               "answers": ANSWERS,
               "stream": len(s.gen), "evictions": s.evictions,
               "vol": [{"name": n, "tok": t, "sec": d} for n, t, d, _, _ in vol_stats]},
              open("composite5_results.json", "w"), indent=1)
    print("COMPOSITE5_DONE")


if __name__ == "__main__":
    main()
