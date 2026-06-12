"""Is the v2 C2/C3 flip after the v7 fixes systematic or variance? The muffin->recompute->
change chain at 3 fresh seeds.  Run next to fft_hf/:  python3 command_rate_test.py"""
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
    h = [0, 0, 0]
    for seed in (5, 6, 7):
        mem = mc.TieredMemory("/dev/null", bge=bge)
        s = AppSession(llm, tok, pooler, bge, iclf, sclf, mem, seed=seed)
        a1, _, _ = s.turn("A bakery sells muffins for $4 each. Maria buys 6 muffins. "
                          "How much does she spend in total?", store="none")
        a2, _, _ = s.turn("Add 2 more muffins and recompute the total.", store="none")
        a3, _, _ = s.turn("I pay with a $50 bill. How much change do I get back?", store="none")
        r = ["24" in a1.replace(",", ""), "32" in a2.replace(",", ""),
             "18" in a3.replace(",", "")]
        for i, v in enumerate(r):
            h[i] += v
        print(f"seed {seed}: total {'HIT' if r[0] else 'MISS'} | recompute "
              f"{'HIT' if r[1] else 'MISS'} ({a2[:60]!r}) | change {'HIT' if r[2] else 'MISS'} "
              f"({a3[:50]!r})", flush=True)
    print(f"RATE: total {h[0]}/3, recompute {h[1]}/3, change {h[2]}/3")
    print("COMMAND_RATE_DONE")


if __name__ == "__main__":
    main()
