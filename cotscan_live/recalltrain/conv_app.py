"""conv_app.py — qualitative app-quality transcript on the faithful runtime with block
recall ON (qk) and the recall-aware adapter merged in. Drives the FULL AppSession.turn()
pipeline (persona, memory gate, decode policy, calculator), --no-web, and prints a
transcript: chitchat (no refusal), a known fact offline, plant personal facts -> distract
(force eviction) -> recall exact values, a math turn. Run from a dir with fft_hf/ + fft_out/.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.getcwd(), "runtime"))

SCRIPT = [
    "Hi! I just got back from a few days in Kyoto, it was lovely.",
    "What's the capital of France?",
    "Please remember that my flight home is JL412 and my locker number is C12.",
    "The weather there was mild and the temples were peaceful.",
    "We walked along the Philosopher's Path one morning.",
    "Tried some great matcha near Gion in the afternoon.",
    "The train back was smooth and on time, thankfully.",
    "I should sort out my packing tonight.",
    "What was my flight number?",
    "And which locker did I use?",
    "What's 18 times 7?",
]

def main():
    import torch, joblib
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from rag import BGERetriever
    import memory_core as mc
    from app_session_torch import AppSession
    from attn_export3_torch import load_pooler
    from block_recall import BlockArchive
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("conv_app: loading ...", flush=True)
    tok = AutoTokenizer.from_pretrained("fft_hf")
    try: llm = AutoModelForCausalLM.from_pretrained("fft_hf", torch_dtype=torch.float32).eval()
    except TypeError: llm = AutoModelForCausalLM.from_pretrained("fft_hf", dtype=torch.float32).eval()
    pooler, bge = load_pooler(), BGERetriever()
    iclf = joblib.load("evals/intent_clf.joblib"); sclf = joblib.load("evals/specificity_clf.joblib")
    mem = mc.TieredMemory("/tmp/conv_app_mem.jsonl", bge=bge)
    mem.session, mem.persistent, mem.pins = [], [], []
    arch = BlockArchive(llm, tok, mode="qk", bge=bge)
    os.environ["SPCHAT_DEVICE"] = dev
    s = AppSession(llm, tok, pooler, bge, iclf, sclf, mem, web=None,
                   rw=384, cap=96, seed=0, profile=True, archive=arch, recall_k=2)
    out = []
    for u in SCRIPT:
        t0 = time.time()
        try:
            r = s.turn(u)
            ans, src = (r[0], r[1]) if isinstance(r, tuple) else (r, "?")
        except Exception as e:
            ans, src = f"<ERROR {type(e).__name__}: {e}>", "err"
        line = f"\nUSER: {u}\n  [{src} {time.time()-t0:.1f}s] {str(ans)[:300]}"
        print(line, flush=True); out.append(line)
    open("conv_app.log", "w").write("\n".join(out))
    print("\nCONV_APP_DONE", flush=True)

if __name__ == "__main__":
    main()
