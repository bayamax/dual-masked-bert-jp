"""Composite battery v3 — correction chains, entity disambiguation, cross-session
recency, math on corrected values, interrogative store-requests, chitchat interleave.

  A  persisted fact corrected in a LATER session (L2-old vs L1-new recency)
  B  two people, similar attributes — recall must not cross the streams
  C  correction-of-correction chain; math on a corrected budget
  D  store-request phrased as a question; chitchat between facts must not leak or log

Run next to fft_hf/:  python3 composite_test3.py
"""
import json, os, re, sys, time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "runtime"))
import memory_core as mc
from app_session_torch import AppSession

PERSIST = "/tmp/composite3_mem.jsonl"
results = []


def run(sess, msg, store, name, want_tier=None, want=None, forbid=None, custom=None):
    t0 = time.time()
    ans, src, chunks = sess.turn(msg, store=store)
    checks = {}
    if want_tier is not None:
        checks["tier"] = (src or "none").startswith(want_tier)
    if want is not None:
        checks["answer"] = all(w.lower() in ans.lower() for w in want)
    if forbid is not None:
        checks["clean"] = all(f.lower() not in ans.lower() for f in forbid)
    if custom is not None:
        checks["custom"] = custom(ans)
    ok = all(checks.values()) if checks else True
    results.append((name, ok, checks))
    print(f"[{name}] {'PASS' if ok else 'FAIL'} {checks} ({time.time() - t0:.0f}s)\n"
          f"   src={src} ans={ans[:140]!r}", flush=True)
    return ans


def main():
    torch.set_num_threads(os.cpu_count())
    import joblib
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from rag import BGERetriever
    sys.path.pop(1)
    from attn_export3_torch import load_pooler

    tok = AutoTokenizer.from_pretrained("fft_hf")
    llm = AutoModelForCausalLM.from_pretrained("fft_hf", dtype=torch.float32).eval()
    pooler, bge = load_pooler(), BGERetriever()
    iclf = joblib.load("evals/intent_clf.joblib")
    sclf = joblib.load("evals/specificity_clf.joblib")
    if os.path.exists(PERSIST):
        os.remove(PERSIST)

    def new_session(seed=0):
        mem = mc.TieredMemory(PERSIST, bge=bge)
        return AppSession(llm, tok, pooler, bge, iclf, sclf, mem, web=None, seed=seed)

    print("#### A — cross-session correction recency ####", flush=True)
    a1 = new_session(seed=21)
    run(a1, "Please remember my parking spot: level B3, spot 47.", "persist", "A1.persist")
    a2 = new_session(seed=22)                       # fresh process: A1 is now L2 (older)
    run(a2, "I moved the car — it's now on level C2, spot 15.", "session", "A2.correction")
    run(a2, "Where is my car parked now?", "none", "A3.recency-recall",
        want=["C2", "15"], forbid=["B3", "47"])

    print("\n#### B — entity disambiguation ####", flush=True)
    b = new_session(seed=23)
    run(b, "My sister Mei lives in Osaka.", "session", "B1.fact")
    run(b, "My colleague Daniel lives in Sapporo.", "session", "B2.fact")
    run(b, "Where does my sister live?", "none", "B3.sister",
        want=["Osaka"], forbid=["Sapporo"])
    run(b, "Where does Daniel live?", "none", "B4.daniel",
        want=["Sapporo"], forbid=["Osaka"])

    print("\n#### C — correction chains + math on corrected value ####", flush=True)
    c = new_session(seed=24)
    run(c, "The team dinner venue is downtown.", "session", "C1.fact")
    run(c, "Actually the venue moved to Shibuya.", "session", "C2.correction")
    run(c, "No wait — it moved to Ginza instead.", "session", "C3.correction2")
    run(c, "Where is the team dinner venue?", "none", "C4.chain-recall",
        want=["Ginza"], forbid=["Shibuya"])
    run(c, "My gift budget is $500.", "session", "C5.fact")
    run(c, "Correction: the budget is $650.", "session", "C6.correction")
    run(c, "If I spend $200 on gifts, how much of my budget is left?", "none",
        "C7.math-corrected", want=["450"], forbid=["300"])

    print("\n#### D — interrogative store-request + chitchat interleave ####", flush=True)
    d = new_session(seed=25)
    run(d, "Can you remember that my locker code is 8042?", "session", "D1.question-store",
        custom=lambda a: "saved" in a.lower() or "got it" in a.lower())
    run(d, "Tell me something interesting about octopuses.", "none", "D2.chitchat",
        forbid=["8042"])
    results.append(("D2.hygiene", all("octopus" not in t.lower() for t in d.mem.session), {}))
    print(f"[D2.hygiene] {'PASS' if results[-1][1] else 'FAIL'} session={d.mem.session}", flush=True)
    run(d, "What's my locker code?", "none", "D3.code-recall", want=["8042"])

    print("\n" + "=" * 70, flush=True)
    for name, ok, _ in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"\nCOMPOSITE3: {sum(1 for _, ok, _ in results if ok)}/{len(results)} PASS")
    json.dump([{"name": n, "ok": o} for n, o, _ in results], open("composite3_results.json", "w"), indent=1)
    print("COMPOSITE3_DONE")


if __name__ == "__main__":
    main()
