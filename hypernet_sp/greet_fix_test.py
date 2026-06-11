"""Post-fix greeting verification (defect #16): bare 'Hello' through the canon SP
pipeline, 3 seeds — must greet, not invent tasks.  Run next to fft_hf/."""
import os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "runtime"))
import memory_core as mc
from app_session_torch import AppSession

GREET = ("hello", "hi", "assist", "help you", "how are", "nice", "welcome", "good ")


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
    ok = 0
    for seed in (71, 72, 73):
        mem = mc.TieredMemory("/dev/null", bge=bge)
        s = AppSession(llm, tok, pooler, bge, iclf, sclf, mem, seed=seed)
        ans, _, _ = s.turn("Hello", store="none")
        good = any(g in ans.lower() for g in GREET) and len(ans) < 400
        ok += good
        print(f"seed {seed}: {'GOOD' if good else 'BAD'} | {ans[:110]!r}", flush=True)
    print(f"GREETING_FIX: {ok}/3")
    print("GREET_FIX_DONE")


if __name__ == "__main__":
    main()
