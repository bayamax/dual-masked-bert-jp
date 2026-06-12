"""Composite battery v6 — WITHIN-SESSION conversational flow: casual turns must follow
the thread of PREVIOUS turns, including turns that never generated (fact acks, recall
quotes). The reported gap: "I went to Kyoto today" -> instant ack left no trace in the
SP/raw stream, so "what do you think was the highlight?" had nothing to follow.

Mechanism under test: turn-stitching (_stitch) — every non-generating turn appends its
User/Assistant exchange to the conversation stream (tokens only, no generation).
Regression guards: the v1 contamination checks (a stitched fact line must not bleed into
an unrelated answer) and math sanity.

Run next to fft_hf/:  python3 composite_test6.py
"""
import json, os, sys, time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "runtime"))
import memory_core as mc
from app_session_torch import AppSession

PERSIST = "/tmp/c6_mem.jsonl"
results = []


def run(sess, msg, store, name, want_any=None, forbid=None, custom=None):
    t0 = time.time()
    ans, src, chunks = sess.turn(msg, store=store)
    checks = {}
    if want_any is not None:
        checks["follows"] = any(w.lower() in ans.lower() for w in want_any)
    if forbid is not None:
        checks["clean"] = all(f.lower() not in ans.lower() for f in forbid)
    if custom is not None:
        checks["custom"] = custom(ans)
    ok = all(checks.values()) if checks else True
    results.append((name, ok, checks))
    print(f"[{name}] {'PASS' if ok else 'FAIL'} {checks} ({time.time() - t0:.0f}s, "
          f"stream={len(sess.gen)})\n   src={src} ans={ans[:160]!r}", flush=True)
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
    mem = mc.TieredMemory(PERSIST, bge=bge)
    s = AppSession(llm, tok, pooler, bge, iclf, sclf, mem, seed=61)

    print("#### A — chitchat follows a FACT turn (the reported gap) ####", flush=True)
    run(s, "I went to Kyoto over the weekend.", "session", "A1.fact-ack")
    assert len(s.gen) > 0, "fact turn left no trace in the stream"
    run(s, "What do you think was probably the highlight of my trip?", "none",
        "A2.follows-fact", want_any=["kyoto", "temple", "shrine", "trip", "kinkaku", "garden"])

    print("\n#### B — chitchat chain follows chitchat ####", flush=True)
    run(s, "I'm thinking about picking up the guitar as a hobby.", "none", "B1.chitchat")
    run(s, "Which one of those would be easiest to start with?", "none",
        "B2.follows-chitchat", want_any=["guitar", "chord", "song", "acoustic", "beginner",
                                         "practice", "start"])

    print("\n#### C — recall quote leaves a trace; contamination must NOT regress ####", flush=True)
    run(s, "My hotel room number was 1408 by the way.", "session", "C1.fact-ack")
    run(s, "What was my room number again?", "none", "C2.recall", want_any=["1408"])
    run(s, "Now explain briefly what a binary search is.", "none", "C3.no-contamination",
        forbid=["1408", "kyoto"],
        want_any=["sorted", "half", "middle", "search", "divide"])

    print("\n#### D — math sanity with a stitched-up stream ####", flush=True)
    run(s, "What is 12 multiplied by 8?", "none", "D1.math", want_any=["96"])

    print("\n" + "=" * 70, flush=True)
    for name, ok, _ in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"\nCOMPOSITE6: {sum(1 for _, ok, _ in results if ok)}/{len(results)} PASS")
    json.dump([{"name": n, "ok": o} for n, o, _ in results], open("composite6_results.json", "w"), indent=1)
    print("COMPOSITE6_DONE")


if __name__ == "__main__":
    main()
