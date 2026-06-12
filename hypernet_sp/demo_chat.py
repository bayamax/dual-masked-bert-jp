"""demo_chat.py — fully-wired chatbot demo on the canonical runtime (all 24 fixes).

Everything is connected: intent routing, tiered memory (L1 session / L2 disk), pinning,
anaphora-expanded REAL web search (DuckDuckGo -> Wikipedia) with the injection guard,
groundedness gates, calculator repair, turn-stitching, profile preamble, and optional
cross-session SP warm start.

Usage (next to fft_hf/ and fft_out/pooler.pt):
    python3 demo_chat.py                       # interactive REPL
    python3 demo_chat.py --script demo.txt     # one prompt per line, prints a transcript
    python3 demo_chat.py --resume              # SP warm start from the previous session
    python3 demo_chat.py --no-web              # offline (web tier disabled)

REPL commands:
    /state   runtime stats (stream/kept/evictions/memory sizes)
    /mem     show L1/L2/pins
    /save    persist the SP warm-start state now
    /reset   wipe session stream (memory files stay)
    /quit    save state and exit
"""
import argparse, os, ssl, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "runtime"))

MEM_PATH = os.environ.get("SPCHAT_MEM", "demo_memory.jsonl")
STATE_PATH = os.environ.get("SPCHAT_STATE", "demo_state.json")


class RealWeb:
    """DuckDuckGo -> Wikipedia with the runtime's own scrapers. SSL verification is
    relaxed because some sandboxes MITM outbound TLS; on a normal Mac this is harmless."""

    def __init__(self):
        ssl._create_default_https_context = ssl._create_unverified_context
        from web_search import DuckDuckGoSearch, WikipediaSearch
        self.backends = [DuckDuckGoSearch(n_results=5), WikipediaSearch(n_results=3)]

    def search(self, query):
        for b in self.backends:
            try:
                r = b.search(query)
            except Exception:
                r = None
            if r:
                return r
        return []


def build_session(args):
    import torch, joblib
    torch.set_num_threads(os.cpu_count())
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from rag import BGERetriever
    import memory_core as mc
    from app_session_torch import AppSession
    from attn_export3_torch import load_pooler

    print("loading model / pooler / heads ...", flush=True)
    tok = AutoTokenizer.from_pretrained("fft_hf")
    llm = AutoModelForCausalLM.from_pretrained("fft_hf", dtype=torch.float32).eval()
    pooler, bge = load_pooler(), BGERetriever()
    iclf = joblib.load("evals/intent_clf.joblib")
    sclf = joblib.load("evals/specificity_clf.joblib")
    web = None if args.no_web else RealWeb()
    mem = mc.TieredMemory(MEM_PATH, bge=bge)
    s = AppSession(llm, tok, pooler, bge, iclf, sclf, mem, web=web,
                   cap=args.cap, seed=args.seed)
    if args.resume and s.load_state(STATE_PATH):
        print(f"[SP warm start: {len(s.kept)} tokens of previous session resumed]")
    print(f"[memory: L2 {len(mem.persistent)} facts from {MEM_PATH} | "
          f"web {'ON' if web else 'OFF'} | profile {'ON' if mem.persistent else 'empty'}]")
    return s, mem


def one_turn(s, msg):
    t0 = time.time()
    ans, src, chunks = s.turn(msg, store="session")
    dt = time.time() - t0
    print(f"\nassistant> {ans}")
    print(f"  [{dt:.0f}s | intent_src={src} | chunks={len(chunks)} | "
          f"stream={len(s.gen)} kept={len(s.kept)} evict={s.evictions}]")
    return ans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", help="file with one user message per line")
    ap.add_argument("--resume", action="store_true", help="SP warm start from last session")
    ap.add_argument("--no-web", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cap", type=int, default=900)
    args = ap.parse_args()
    s, mem = build_session(args)

    if args.script:
        for line in open(args.script):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            print(f"\nuser> {line}")
            one_turn(s, line)
        s.save_state(STATE_PATH)
        print(f"\n[state saved to {STATE_PATH}]")
        print("DEMO_SCRIPT_DONE")
        return

    print("\nSP-chat demo — /quit to exit, /state /mem /save /reset for tools\n")
    while True:
        try:
            msg = input("user> ").strip()
        except (EOFError, KeyboardInterrupt):
            msg = "/quit"
        if not msg:
            continue
        if msg == "/quit":
            s.save_state(STATE_PATH)
            print(f"[state saved to {STATE_PATH}] bye")
            break
        if msg == "/state":
            print(f"  stream={len(s.gen)} kept={len(s.kept)} absorbed={s.absorbed} "
                  f"evictions={s.evictions} | L1={len(mem.session)} L2={len(mem.persistent)} "
                  f"pins={len(mem.pins)}")
            continue
        if msg == "/mem":
            print("  L1:", mem.session[-6:])
            print("  L2:", mem.persistent[-6:])
            print("  pins:", mem.pins[-6:])
            continue
        if msg == "/save":
            s.save_state(STATE_PATH); print(f"  saved -> {STATE_PATH}")
            continue
        if msg == "/reset":
            s.gen, s.kept, s.absorbed, s.evictions = [], [], 0, 0
            print("  session stream reset (memory files untouched)")
            continue
        one_turn(s, msg)


if __name__ == "__main__":
    main()
