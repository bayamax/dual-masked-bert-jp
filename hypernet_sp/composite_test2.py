"""Composite battery v2 — adversarial app-usage patterns the v1 battery didn't touch:

  A  corrections-win + multi-fact recall + verbatim letter-code recall
  B  cross-tier shadowing (world question after a related personal fact) and
     UNANSWERABLE personal recall (never-stated fact -> must not confabulate)
  C  3-step chained math: total -> command recompute -> change off the LATEST result
  D  deep recall after distractor facts

Same real stack as v1 (fft_hf, pooler, BGE, shipped heads, canned web).
Run next to fft_hf/:  python3 composite_test2.py
"""
import json, os, re, sys, time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "runtime"))
import memory_core as mc
from app_session_torch import AppSession

PERSIST = "/tmp/composite2_mem.jsonl"


class FakeWeb:
    CORPUS = [
        ("emperor japan naruhito", "Naruhito is the current Emperor of Japan, having acceded to the throne in 2019."),
        ("mount fuji", "Mount Fuji is the highest mountain in Japan, with a summit elevation of 3,776.24 m."),
        ("kyoto temple", "Kyoto's most celebrated temples include Kinkaku-ji and Kiyomizu-dera."),
    ]

    def __init__(self):
        self.queries = []

    def search(self, query):
        self.queries.append(query)
        q = set(re.findall(r"[a-z0-9]+", query.lower()))
        scored = sorted(self.CORPUS, key=lambda kv: -len(q & set(kv[0].split() + kv[1].lower().split())))
        return [t for _, t in scored[:3]]


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


def no_confabulated_secret(ans):
    """An unanswerable recall must not invent a value: no digit-bearing or password-shaped
    token, or an explicit don't-know. ('I don't have that saved' passes; 'hunter2' fails)."""
    refusal = re.search(r"don'?t (have|know)|not (stored|saved|mentioned|provided)|no (record|information)|"
                        r"haven'?t (told|shared|mentioned)|unknown", ans, re.I)
    invented = re.search(r"\b[A-Za-z]*\d[A-Za-z0-9\-]{2,}\b|\bpassword is\b", ans, re.I)
    return bool(refusal) or not invented


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
    web = FakeWeb()

    def new_session(seed=0):
        mem = mc.TieredMemory(PERSIST, bge=bge)
        return AppSession(llm, tok, pooler, bge, iclf, sclf, mem, web=web, seed=seed)

    print("#### A — corrections, multi-fact, letter codes ####", flush=True)
    a = new_session(seed=11)
    run(a, "I'll paint the bookshelf blue with 5 shelves.", "session", "A1.fact")
    run(a, "Actually, make it 6 shelves instead of 5.", "session", "A2.correction")
    run(a, "And the color should be matte black, not blue.", "session", "A3.correction")
    run(a, "What color am I painting it and how many shelves now?", "none", "A4.multi-fact-recall",
        want_tier="L1", want=["black", "6"])
    run(a, "My gym locker code is QX7-2291.", "session", "A5.fact-code")
    run(a, "What's my gym locker code?", "none", "A6.code-recall",
        want_tier="L1", want=["QX7-2291"])

    print("\n#### B — shadowing & unanswerable recall ####", flush=True)
    b = new_session(seed=12)
    run(b, "I'm planning a trip to Kyoto next month.", "session", "B1.fact")
    run(b, "Who is the current emperor of Japan?", "none", "B2.world-not-shadowed",
        want_tier="L3", want=["Naruhito"])
    run(b, "What's my wifi password?", "none", "B3.unanswerable",
        custom=no_confabulated_secret)

    print("\n#### C — chained math with command recompute ####", flush=True)
    c = new_session(seed=13)
    run(c, "A bakery sells muffins for $4 each. Maria buys 6 muffins. How much does she spend in total?",
        "none", "C1.math", want=["24"])
    run(c, "Add 2 more muffins and recompute the total.", "none", "C2.command-recompute",
        want=["32"])
    run(c, "I pay with a $50 bill. How much change do I get back?", "none", "C3.change-latest",
        want=["18"])

    print("\n#### D — deep recall after distractor facts ####", flush=True)
    d = new_session(seed=14)
    run(d, "My dentist appointment is on the 15th at 9am.", "session", "D1.fact")
    run(d, "My sister's dog is named Mochi.", "session", "D2.distractor")
    run(d, "The project deadline moved to next Friday.", "session", "D3.distractor")
    run(d, "I parked on level B3, spot 47 today.", "session", "D4.distractor")
    run(d, "When is my dentist appointment?", "none", "D5.deep-recall",
        want_tier="L1", want=["15th", "9"])

    print("\n" + "=" * 70, flush=True)
    for name, ok, _ in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"\nCOMPOSITE2: {sum(1 for _, ok, _ in results if ok)}/{len(results)} PASS")
    json.dump([{"name": n, "ok": o} for n, o, _ in results], open("composite2_results.json", "w"), indent=1)
    print("COMPOSITE2_DONE")


if __name__ == "__main__":
    main()
