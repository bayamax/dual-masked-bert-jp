"""Composite battery v4 — MARATHON: one ~40-turn session simulating a day of app use,
then harvest-phase probes against everything that accumulated.

Stress targets (none covered by v1-v3, all core to the bounded-memory promise):
  * early-fact survival: facts from turns 1-8 recalled 25+ turns later
  * pin-buffer rotation: more specific values than PINCAP=12 can hold
  * accumulated corrections: parking corrected twice, meeting moved, budget updated
  * SP-stream growth: generation turns interleave so the kept-buffer/SP is genuinely fed
  * latency flatness across the session (bounded KV -> per-turn cost must not trend up)
  * late honest-miss and late world-lookup non-shadowing

Run next to fft_hf/:  python3 composite_test4.py
"""
import json, os, re, sys, time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "runtime"))
import memory_core as mc
from app_session_torch import AppSession

PERSIST = "/tmp/composite4_mem.jsonl"
results, gen_times = [], []


class FakeWeb:
    CORPUS = [("everest", "Mount Everest is Earth's highest mountain, elevation 8,848.86 m."),
              ("tokyo tower", "Tokyo Tower is 333 m tall, completed in 1958.")]

    def search(self, query):
        q = set(re.findall(r"[a-z0-9]+", query.lower()))
        scored = sorted(self.CORPUS, key=lambda kv: -len(q & set(kv[0].split() + kv[1].lower().split())))
        return [t for _, t in scored[:3]]


def run(sess, msg, store, name, want_tier=None, want=None, forbid=None, custom=None, is_gen=False):
    t0 = time.time()
    ans, src, chunks = sess.turn(msg, store=store)
    dt = time.time() - t0
    if is_gen:
        gen_times.append((name, dt))
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
    print(f"[{name}] {'PASS' if ok else 'FAIL'} {checks} ({dt:.0f}s)\n"
          f"   src={src} ans={ans[:130]!r}", flush=True)
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
    s = AppSession(llm, tok, pooler, bge, iclf, sclf, mem, web=FakeWeb(), seed=31)

    print("#### phase 1 (turns 1-8): morning setup — facts ####", flush=True)
    for i, f in enumerate([
            "My name is Aki Tanaka.",                                   # t1
            "My employee ID is EMP-90832.",                             # t2
            "I'm allergic to peanuts.",                                 # t3
            "I parked on level B3, spot 47.",                           # t4
            "The project is called Apollo, deadline next Friday.",      # t5
            "My gift budget is $500.",                                  # t6
            "The team meeting is at 2pm.",                              # t7
            "My hotel tonight is the Grand Palace, room 1408."], 1):    # t8
        run(s, f, "session", f"t{i}.fact")

    print("\n#### phase 2 (turns 9-12): work — math + followup ####", flush=True)
    run(s, "A vendor quote is $120 per unit for 8 units. What's the total?", "none",
        "t9.math", want=["960"], is_gen=True)
    run(s, "We get a 10% discount on that. What's the final price?", "none",
        "t10.math-follow", want=["864"], is_gen=True)
    run(s, "Thanks! What should I check before signing a vendor contract?", "none",
        "t11.chitchat", forbid=["EMP-90832", "1408"], is_gen=True)
    run(s, "What's my employee ID?", "none", "t12.early-recall", want=["EMP-90832"])

    print("\n#### phase 3 (turns 13-22): afternoon — corrections + pin pressure ####", flush=True)
    run(s, "I moved the car — now it's level C2, spot 15.", "session", "t13.correction")
    run(s, "The meeting moved to 4:30pm.", "session", "t14.correction")
    run(s, "Budget update: it's $650 now.", "session", "t15.correction")
    for i, f in enumerate([
            "My visitor badge code is VB-7731.",
            "The wifi password here is k9x2m4.",
            "My lunch order number is 88.",
            "The printer access pin is 5512.",
            "Conference room is 12F-B.",
            "My taxi reservation is TX-4419.",
            "The client's name is Ms. Watanabe."], 16):
        run(s, f, "session", f"t{i}.fact")                              # pins rotate past 12

    print("\n#### phase 4 (turns 23-26): evening — distraction + world lookup ####", flush=True)
    run(s, "Recommend a relaxing thing to do after work.", "none",
        "t23.chitchat", forbid=["8042", "EMP-90832", "k9x2m4"], is_gen=True)
    run(s, "How tall is Tokyo Tower?", "none", "t24.lookup",
        want_tier="L3", want=["333"])
    run(s, "Actually scratch the car move — it's back at B3, spot 47.", "session", "t25.correction2")
    run(s, "If I spend $200 from my budget, how much is left?", "none",
        "t26.math-corrected", want=["450"], is_gen=True)

    print("\n#### phase 5 (turns 27-36): HARVEST — recall everything ####", flush=True)
    run(s, "What's my name?", "none", "t27.name", want=["Aki"])
    run(s, "Am I allergic to anything?", "none", "t28.allergy", want=["peanut"])
    run(s, "Where is my car parked now?", "none", "t29.parking-final",
        want=["B3", "47"], forbid=["C2"])
    run(s, "What time is the meeting now?", "none", "t30.meeting", want=["4:30"])
    run(s, "What's the project called and when is the deadline?", "none",
        "t31.multi-fact", want=["Apollo", "Friday"])
    run(s, "What's my hotel room number?", "none", "t32.hotel", want=["1408"])
    run(s, "What's the wifi password here?", "none", "t33.mid-fact", want=["k9x2m4"])
    run(s, "Who is the client again?", "none", "t34.client", want=["Watanabe"])
    run(s, "What's my blood type?", "none", "t35.honest-miss",
        custom=lambda a: bool(re.search(r"don'?t have|haven'?t told|not? (saved|record)", a, re.I)))
    run(s, "What's my visitor badge code?", "none", "t36.badge", want=["VB-7731"])

    print("\n" + "=" * 70, flush=True)
    for name, ok, _ in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print("\nlatency of generation turns across the session (bounded KV -> should be ~flat):")
    for name, dt in gen_times:
        print(f"  {name}: {dt:.0f}s")
    print(f"session log size: {len(mem.session)} | pins: {len(mem.pins)}/12 | gen stream: {len(s.gen)} tok "
          f"| absorbed into SP buffer: {s.absorbed} tok")
    print(f"\nCOMPOSITE4: {sum(1 for _, ok, _ in results if ok)}/{len(results)} PASS")
    json.dump([{"name": n, "ok": o} for n, o, _ in results], open("composite4_results.json", "w"), indent=1)
    print("COMPOSITE4_DONE")


if __name__ == "__main__":
    main()
