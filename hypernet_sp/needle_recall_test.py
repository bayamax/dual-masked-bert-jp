"""needle_recall_test.py — Phase A acceptance test (BLOCKRECALL_V1 harness).

Question: after casual mentions scroll out of the exposure window (so they exist only as
SP gist), can verbatim block re-injection restore exact-value recall — and does scoring
in the model's OWN Q/K space beat the same payload scored in BGE text-embedding space?
This reproduces the EM-LLM vs external-embedder comparison inside our runtime.

Conditions (same questions, same preloaded history, _gen_once called directly so the
memory layer / routing never interferes — this isolates the generation-context mechanism):
  sp    : status quo — SP(32) + raw window 512, no recall
  qk    : + BlockArchive recall, model Q/K space scoring     (Phase A proposal)
  bge   : + BlockArchive recall, BGE cosine scoring          (same token budget)
  full  : raw window 8192 — the full-KV ceiling

Score: the needle's exact value appears in the extracted answer.
Run from the directory containing fft_hf/ and fft_out/:
    python3 hypernet_sp/needle_recall_test.py [--smoke]
"""
import argparse, copy, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.getcwd(), "runtime"))   # rag.py / web_search.py

NEEDLES = [
    ("By the way, the wifi password at the mountain cabin was trout-9182. Took us forever to find it.",
     "What was the wifi password at the mountain cabin?", "9182"),
    ("I finally reached Dr. Imai today - her office extension is 4408.",
     "What is Dr. Imai's office extension?", "4408"),
    ("The rover prototype came in at 18.4 kg on the bench scale this morning.",
     "How much did the rover prototype weigh?", "18.4"),
    ("My cousin finally named his fishing boat: the Yume-Maru.",
     "What is my cousin's fishing boat called?", "yume"),
    ("Parking at the venue was a maze - we ended up in spot B-37.",
     "Which parking spot did we end up in at the venue?", "37"),
    ("That limited edition sake was 6800 yen, totally worth it though.",
     "How much did the limited edition sake cost?", "6800"),
    ("The night train to Aomori leaves at 23:46, I double checked.",
     "What time does the night train to Aomori leave?", "23:46"),
    ("The projector in conference room A is an XV-310, in case anyone asks.",
     "What model is the projector in conference room A?", "310"),
    ("Grandma's bread recipe needs exactly 320 grams of flour, she was very firm about that.",
     "How many grams of flour does grandma's bread recipe need?", "320"),
    ("We turned back at trail marker 58 because of the fog.",
     "At which trail marker did we turn back?", "58"),
]

FILLER = [
    ("The weather this week has been all over the place, sunny then storms.",
     "It really has. Those sudden shifts make it hard to plan anything outdoors, though the cool evenings after the rain have been pleasant for walks."),
    ("I tried that new ramen place near the station.",
     "How was it? A good bowl of ramen after a long day is hard to beat - the broth makes or breaks it in my experience."),
    ("Work has been busy but manageable lately.",
     "That sounds like a decent balance. Busy enough to stay engaged without burning out is the sweet spot most people aim for."),
    ("I started reading a novel about Antarctic explorers.",
     "Polar exploration stories can be gripping - the isolation and the small daily decisions that decide everything give them real tension."),
    ("My neighbor's dog learned to open the gate somehow.",
     "Clever dog! Once they figure out latches there is no stopping them - you may need a second clip higher up where paws cannot reach."),
    ("We are thinking about repainting the kitchen.",
     "A fresh coat changes the whole feel of a room. Lighter colors tend to make small kitchens feel bigger, especially with warm lighting."),
    ("The train was delayed twenty minutes this morning.",
     "Annoying way to start the day. At least with a book or some music the platform wait goes a little faster."),
    ("I have been trying to drink more water instead of coffee.",
     "A solid habit. Keeping a full bottle on the desk where you can see it does most of the work, honestly."),
    ("The garden tomatoes are finally turning red.",
     "Nothing beats a sun-warmed tomato straight off the vine. The first ripe one of the season always feels like a small victory."),
    ("I watched a documentary about deep sea creatures last night.",
     "The deep sea is wilder than fiction - bioluminescent lures, pressure-proof bodies, whole ecosystems around vents. What stuck with you most?"),
    ("My phone battery barely lasts the afternoon now.",
     "Batteries do fade after a couple of years. Lowering screen brightness and trimming background apps can buy you a few more months."),
    ("We played board games with friends over the weekend.",
     "Board game nights are underrated. The slow conversational pace between turns is half the fun, win or lose."),
    ("There was a small earthquake here yesterday, nothing serious.",
     "Glad it was minor. Always a good prompt to check the emergency kit and strap down the tall furniture, just in case."),
    ("I am learning to make sourdough bread.",
     "A rewarding rabbit hole. Keeping the starter happy is the hard part - consistent feeding times matter more than fancy flour."),
    ("The library extended its evening hours this month.",
     "That is great news for anyone who works late. Quiet evening hours at a library are some of the best focused time there is."),
    ("My bicycle needs a new chain, it keeps slipping.",
     "A slipping chain usually means it has stretched past its service life. Replacing it early also saves the cassette from extra wear."),
]


def build_history(tok, needles, target=4200):
    """Interleave needle mentions with filler, then pad with more filler until the
    stream reaches `target` tokens — every needle ends up deep in the evicted region,
    far beyond the 512-token exposure window."""
    pairs = []
    fi = 0
    for n_user, _, _ in needles:
        pairs.append(FILLER[fi % len(FILLER)])
        fi += 1
        pairs.append((n_user, "Noted - that sounds like quite a day. Thanks for telling me."))
    ids = []

    def emit(u, a):
        text = ("<｜end▁of▁sentence｜>" if ids else "") + f"<｜User｜>{u}<｜Assistant｜>{a}"
        ids.extend(tok.encode(text, add_special_tokens=False))

    for u, a in pairs:
        emit(u, a)
    lap = 0
    while len(ids) < target:
        u, a = FILLER[fi % len(FILLER)]
        if lap:
            u = f"One more thing about that - {u[0].lower()}{u[1:]}"
        emit(u, a)
        fi += 1
        if fi % len(FILLER) == 0:
            lap += 1
    return ids


def build(cond, args):
    import torch, joblib
    torch.set_num_threads(os.cpu_count())
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from rag import BGERetriever
    import memory_core as mc
    from app_session_torch import AppSession
    from attn_export3_torch import load_pooler
    from block_recall import BlockArchive
    global _ASSETS
    if "_ASSETS" not in globals():
        print("loading model/pooler/heads ...", flush=True)
        tok = AutoTokenizer.from_pretrained("fft_hf")
        llm = AutoModelForCausalLM.from_pretrained("fft_hf", dtype=torch.float32).eval()
        _ASSETS = (tok, llm, load_pooler(), BGERetriever(),
                   joblib.load("evals/intent_clf.joblib"),
                   joblib.load("evals/specificity_clf.joblib"))
    tok, llm, pooler, bge, iclf, sclf = _ASSETS
    mem = mc.TieredMemory("/tmp/needle_mem.jsonl", bge=bge)
    mem.session, mem.persistent, mem.pins = [], [], []
    rw = 8192 if cond == "full" else 512
    archive = None
    if cond in ("qk", "bge"):
        archive = BlockArchive(llm, tok, mode=cond, bge=bge)
    s = AppSession(llm, tok, pooler, bge, iclf, sclf, mem, web=None,
                   rw=rw, cap=64, seed=0, profile=False, archive=archive,
                   recall_k=args.k)
    return s, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--conds", default="sp,qk,bge,full")
    ap.add_argument("--k", type=int, default=2)
    args = ap.parse_args()
    needles = NEEDLES[:3] if args.smoke else NEEDLES
    results = {}
    print("HARNESS BLOCKRECALL_V1", flush=True)
    for cond in args.conds.split(","):
        s, tok = build(cond, args)
        hist = build_history(tok, needles)
        print(f"\n=== cond={cond} history={len(hist)} tokens ===", flush=True)
        rows = []
        for n_user, q, val in needles:
            s.gen, s.kept, s.absorbed = list(hist), [], 0
            if s.archive is not None:
                s.archive.buf, s.archive.blocks = [], []
            t0 = time.time()
            ans = s._gen_once(q, force_think=False, temp_override=0.05,
                              cap=56, salvage="", salvage_budget=24)
            ok = val.lower() in (ans or "").lower().replace(",", "")
            dt = time.time() - t0
            nb = len(s.archive.blocks) if s.archive is not None else 0
            rows.append({"q": q, "val": val, "ans": (ans or "")[:160], "ok": ok})
            print(f"  [{'OK' if ok else 'MISS'}] {dt:5.0f}s blocks={nb} {q}\n"
                  f"        -> {(ans or '')[:140]}", flush=True)
        acc = sum(r["ok"] for r in rows)
        results[cond] = {"acc": acc, "n": len(rows), "rows": rows}
        print(f"  cond={cond}: {acc}/{len(rows)}", flush=True)
    print("\n=== SUMMARY ===")
    for cond, r in results.items():
        print(f"  {cond:5s} {r['acc']}/{r['n']}")
    with open("needle_results.json", "w") as f:
        json.dump(results, f, indent=1)
    print("NEEDLE_DONE")


if __name__ == "__main__":
    main()
