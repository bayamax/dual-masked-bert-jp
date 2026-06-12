"""Diagnose composite FAIL S2.math-followup (answered 24 instead of 26): rerun the muffin
pair with full think-stream dumps + DecodePolicy firing telemetry, so we can see WHETHER
the convergence trigger fired on the intermediate value (24) and force-closed the think
before the change computation reached 26.   python3 debug_followup.py [seed]"""
import os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "runtime"))
import memory_core as mc
import decode_policy
from app_session_torch import AppSession


def main():
    torch.set_num_threads(os.cpu_count())
    import joblib
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from rag import BGERetriever
    sys.path.pop(1)
    from attn_export3_torch import load_pooler
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1

    fired_log = []
    orig = decode_policy.DecodePolicy.note_text

    def spy(self, text):
        r = orig(self, text)
        if r:
            fired_log.append({"counts": dict(self.counts), "at_chars": len(text),
                              "converged": self.converged_answer()})
        return r
    decode_policy.DecodePolicy.note_text = spy

    tok = AutoTokenizer.from_pretrained("fft_hf")
    llm = AutoModelForCausalLM.from_pretrained("fft_hf", dtype=torch.float32).eval()
    pooler, bge = load_pooler(), BGERetriever()
    mem = mc.TieredMemory("/dev/null", bge=bge)
    mem.session.append("My hotel room number for tonight is 1408.")   # state as in S2
    mem.pin("My hotel room number for tonight is 1408.")
    s = AppSession(llm, tok, pooler, bge, joblib.load("evals/intent_clf.joblib"),
                   joblib.load("evals/specificity_clf.joblib"), mem, seed=seed)

    for msg, want in [
        ("A bakery sells muffins for $4 each. Maria buys 6 muffins. How much does she spend in total?", "24"),
        ("I pay with a $50 bill. How much change do I get back?", "26"),
    ]:
        start = len(s.gen)
        ans, src, chunks = s.turn(msg, store="none")
        body = tok.decode(s.gen[start:])
        print("=" * 70)
        print(f"Q: {msg}\nsrc={src}\nchunks={chunks}\nANSWER: {ans!r}  (want {want}: "
              f"{'HIT' if want in ans else 'MISS'})")
        print(f"policy fired: {fired_log[-1] if fired_log else 'no'}")
        fired_log.clear()
        print(f"--- turn body ({len(s.gen) - start} tok) ---\n{body[:3000]}")
    print("DEBUG_FOLLOWUP_DONE")


if __name__ == "__main__":
    main()
