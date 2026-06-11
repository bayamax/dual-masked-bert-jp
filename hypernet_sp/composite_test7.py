"""Composite battery v7 — GENERAL CHAT: the bread-and-butter use cases of a chat-LLM app
that every earlier battery (memory/math-centric) skipped:

  W  writing/drafting + iterative refinement ("make it shorter")
  B  brainstorm lists
  S  summarisation of pasted text
  R  rewriting (politeness)
  H  how-to instructions
  E  emotional support / venting
  J  jokes + casual follow-up
  K  open knowledge ("why is the sky blue")
  M  memory x creative (poem about the user's cat)

All checks mechanical. Run next to fft_hf/:  python3 composite_test7.py
"""
import json, os, re, sys, time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "runtime"))
import memory_core as mc
from app_session_torch import AppSession

results, ANSWERS = [], {}

PASSAGE = ("The lighthouse keeper Elena spent thirty years on the rocky island of Skellan. "
           "Every evening she climbed the spiral stairs to light the lamp, and every morning "
           "she logged the passing ships in a leather journal. When the light was automated, "
           "she stayed on as caretaker, guiding tourists through the tower and telling them "
           "how the beam once saved a fishing fleet during the great storm. On her last day "
           "she left the journal on the top step, open to the first page, for whoever came next.")


def words(s):
    return len(s.split())


def run(sess, msg, store, name, custom, note=""):
    t0 = time.time()
    ans, src, chunks = sess.turn(msg, store=store)
    ok = bool(custom(ans))
    results.append((name, ok))
    ANSWERS[name] = ans
    print(f"[{name}] {'PASS' if ok else 'FAIL'} ({time.time() - t0:.0f}s, {words(ans)}w, "
          f"intent_src={src})\n   ans={ans[:150]!r}", flush=True)
    return ans


def no_think_leak(a):
    return "<think>" not in a and "</think>" not in a


def main():
    torch.set_num_threads(os.cpu_count())
    import joblib
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from rag import BGERetriever
    sys.path.pop(1)
    from attn_export3_torch import load_pooler
    import app_session_torch
    app_session_torch.ANSCAP = 2000

    tok = AutoTokenizer.from_pretrained("fft_hf")
    llm = AutoModelForCausalLM.from_pretrained("fft_hf", dtype=torch.float32).eval()
    pooler, bge = load_pooler(), BGERetriever()
    iclf = joblib.load("evals/intent_clf.joblib")
    sclf = joblib.load("evals/specificity_clf.joblib")
    mem = mc.TieredMemory("/dev/null", bge=bge)
    s = AppSession(llm, tok, pooler, bge, iclf, sclf, mem, cap=900, seed=91)

    print("#### W — drafting + refinement ####", flush=True)
    w1 = run(s, "Write a short polite email to my landlord asking them to fix the dripping "
                "kitchen tap.", "none", "W1.draft",
             lambda a: ("tap" in a.lower() or "faucet" in a.lower()) and words(a) >= 30
             and no_think_leak(a) and not re.search(r"= ?\d", a))
    run(s, "Make it shorter and friendlier.", "none", "W2.refine",
        lambda a: 0 < words(a) < max(words(w1), 35) and
        ("tap" in a.lower() or "faucet" in a.lower() or "drip" in a.lower())
        and not re.search(r"final number|= ?\d", a.lower()))

    print("\n#### B — brainstorm ####", flush=True)
    run(s, "Give me five dinner ideas with chicken.", "none", "B1.list",
        lambda a: len(re.findall(r"(?:^|\n)\s*(?:\d+[.)]|[-*])\s+", a)) >= 4
        and "chicken" in a.lower())

    print("\n#### S — summarisation of pasted text ####", flush=True)
    run(s, f"Summarize this in two sentences: {PASSAGE}", "none", "S1.summary",
        lambda a: 8 <= words(a) <= 90 and ("elena" in a.lower() or "lighthouse" in a.lower())
        and no_think_leak(a))

    print("\n#### R — rewrite politely ####", flush=True)
    run(s, "Rewrite this to sound more polite: 'Send me the report now.'", "none", "R1.polite",
        lambda a: re.search(r"please|could|would|kindly", a.lower()) and "report" in a.lower())

    print("\n#### H — how-to ####", flush=True)
    run(s, "How can I make my phone battery last longer?", "none", "H1.howto",
        lambda a: words(a) >= 25 and re.search(r"brightness|screen|background|battery|mode",
                                               a.lower()))

    print("\n#### E — emotional ####", flush=True)
    run(s, "I had a really rough day at work and just need to vent for a second.", "none",
        "E1.vent", lambda a: re.search(r"sorry|sounds|tough|rough|hear|here for|understand",
                                       a.lower()) and not re.search(r"= ?\d|theorem", a.lower()))

    print("\n#### J — jokes + casual follow-up ####", flush=True)
    run(s, "Tell me a short joke.", "none", "J1.joke",
        lambda a: 0 < words(a) <= 80 and no_think_leak(a))
    run(s, "Haha, got another one?", "none", "J2.another",
        lambda a: 0 < words(a) <= 100 and not re.search(r"final number|= ?\d", a.lower()))

    print("\n#### K — open knowledge ####", flush=True)
    run(s, "Why is the sky blue?", "none", "K1.sky",
        lambda a: re.search(r"scatter|wavelength|light|rayleigh", a.lower()) and words(a) >= 20)

    print("\n#### M — memory x creative ####", flush=True)
    run(s, "My cat is named Mochi, by the way.", "session", "M1.fact", lambda a: True)
    run(s, "Write a two-line poem about my cat.", "none", "M2.poem",
        lambda a: "mochi" in a.lower() and words(a) <= 60)

    print("\n" + "=" * 70, flush=True)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    npass = sum(1 for _, ok in results if ok)
    print(f"\nCOMPOSITE7: {npass}/{len(results)} PASS")
    json.dump({"results": [{"name": n, "ok": o} for n, o in results], "answers": ANSWERS},
              open("composite7_results.json", "w"), indent=1)
    print("COMPOSITE7_DONE")


if __name__ == "__main__":
    main()
