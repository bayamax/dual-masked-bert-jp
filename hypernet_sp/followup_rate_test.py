"""Is the math-followup flip after defect-16 (no empty-SP) systematic or variance?
The canonical muffin pair at 3 seeds, post-fix. Pre-fix record: single-seed PASS.
Run next to fft_hf/:  python3 followup_rate_test.py"""
import os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "runtime"))
import memory_core as mc
from app_session_torch import AppSession


def main():
    torch.set_num_threads(os.cpu_count())
    import joblib
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from rag import BGERetriever
    sys.path.pop(1)
    from attn_export3_torch import load_pooler
    tok = AutoTokenizer.from_pretrained("fft_hf")
    llm = AutoModelForCausalLM.from_pretrained("fft_hf", dtype=torch.float32).eval()
    bge = BGERetriever()
    iclf = joblib.load("evals/intent_clf.joblib")
    sclf = joblib.load("evals/specificity_clf.joblib")
    pooler = load_pooler()
    hit1 = hit2 = 0
    for seed in (1, 2, 3):
        mem = mc.TieredMemory("/dev/null", bge=bge)
        s = AppSession(llm, tok, pooler, bge, iclf, sclf, mem, seed=seed)
        a1, _, _ = s.turn("A bakery sells muffins for $4 each. Maria buys 6 muffins. "
                          "How much does she spend in total?", store="none")
        h1 = "24" in a1.replace(",", "")
        a2, _, _ = s.turn("I pay with a $50 bill. How much change do I get back?", store="none")
        h2 = "26" in a2.replace(",", "")
        hit1 += h1; hit2 += h2
        print(f"seed {seed}: total {'HIT' if h1 else 'MISS'} ({a1[:60]!r}) | "
              f"change {'HIT' if h2 else 'MISS'} ({a2[:60]!r})", flush=True)
    print(f"RATE: total {hit1}/3, followup {hit2}/3")
    print("FOLLOWUP_RATE_DONE")


if __name__ == "__main__":
    main()
