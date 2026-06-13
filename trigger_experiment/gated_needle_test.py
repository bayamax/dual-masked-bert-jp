"""gated_needle_test.py — wiring acceptance for trigger-gated block recall.

Arms (same preloaded ~4200-token history, bge-mode BlockArchive, _gen_once called
directly so routing never interferes — the needle_recall_test harness shape):
  always : archive recall every turn (status quo; V1 reference 7/10)
  gated  : recall only when the TRIGGER_V3 head fires (thresh -2.6, TPR-first)

Questions:
  10 evicted needles  — gate must FIRE; value recall must match `always`
   4 in-window needles — gate must SKIP; value still answerable from the raw window
   6 generic           — gate must SKIP
Run next to fft_hf/ and fft_out/:  python3 hypernet_sp/gated_needle_test.py [--smoke]
"""
import argparse, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.getcwd(), "runtime"))

from needle_recall_test import FILLER, NEEDLES, build_history

WINDOW_NEEDLES = [
    ("Oh, before I forget - my gym locker code this month is 44-19-72.",
     "What is my gym locker code this month?", "44-19-72"),
    ("The dentist moved my appointment to 15:40 on Thursday.",
     "What time is my dentist appointment on Thursday?", "15:40"),
    ("My new bike lock combination is 8351, by the way.",
     "What is my new bike lock combination?", "8351"),
    ("The package tracking number ends in 77041.",
     "What does the package tracking number end in?", "77041"),
]

GENERIC_QS = [
    "Can you suggest a quick dinner idea for tonight?",
    "What is a good way to stay focused in the afternoon?",
    "Could you recommend a light exercise before bed?",
    "What snack goes well with green tea?",
    "Any tips for falling asleep faster?",
    "How should I store fresh basil so it lasts?",
]


def append_pairs(tok, ids, pairs):
    for u, a in pairs:
        text = "<｜end▁of▁sentence｜>" + f"<｜User｜>{u}<｜Assistant｜>{a}"
        ids.extend(tok.encode(text, add_special_tokens=False))


def build_session():
    import torch
    torch.set_num_threads(os.cpu_count())
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from rag import BGERetriever
    import memory_core as mc
    from app_session_torch import AppSession
    from attn_export3_torch import load_pooler
    from block_recall import BlockArchive
    global _ASSETS
    if "_ASSETS" not in globals():
        print("loading model/pooler/bge ...", flush=True)
        tok = AutoTokenizer.from_pretrained("fft_hf")
        llm = AutoModelForCausalLM.from_pretrained("fft_hf", dtype=torch.float32).eval()
        if os.environ.get("SPCHAT_DEVICE"):
            llm.to(os.environ["SPCHAT_DEVICE"])
        _ASSETS = (tok, llm, load_pooler(), BGERetriever())
    tok, llm, pooler, bge = _ASSETS
    mem = mc.TieredMemory("/tmp/gated_mem.jsonl", bge=bge)
    mem.session, mem.persistent, mem.pins = [], [], []
    archive = BlockArchive(llm, tok, mode="bge", bge=bge)
    s = AppSession(llm, tok, pooler, bge, None, None, mem, web=None,
                   rw=512, cap=64, seed=0, profile=False, archive=archive, recall_k=2)
    return s, tok


def run_arm(name, gate, questions, hist, s, tok):
    rows = []
    for qtype, q, val in questions:
        s.gen, s.kept, s.absorbed = list(hist), [], 0
        s.archive.buf, s.archive.blocks = [], []
        s.gate, s.gate_log = gate, []
        t0 = time.time()
        ans = s._gen_once(q, force_think=False, temp_override=0.05,
                          cap=56, salvage="", salvage_budget=24)
        fired = s.gate_log[0][1] if s.gate_log else (gate is None)
        score = s.gate_log[0][2] if s.gate_log else None
        ok = (val is None) or (val.lower() in (ans or "").lower().replace(",", ""))
        rows.append({"qtype": qtype, "q": q, "val": val, "ans": (ans or "")[:160],
                     "ok": ok, "fired": bool(fired), "score": score})
        print(f"  [{name}] {qtype:7s} fired={str(fired):5s} score={score} "
              f"{'OK ' if ok else 'MISS'} ({time.time()-t0:4.0f}s) {q[:48]}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    s, tok = build_session()
    needles = NEEDLES[:3] if args.smoke else NEEDLES
    hist = build_history(tok, needles)
    win = WINDOW_NEEDLES[:1] if args.smoke else WINDOW_NEEDLES
    gen = GENERIC_QS[:1] if args.smoke else GENERIC_QS
    append_pairs(tok, hist, [(u, "Noted - thanks for telling me.") for u, _q, _v in win])
    questions = ([("evict", q, v) for _u, q, v in needles]
                 + [("window", q, v) for _u, q, v in win]
                 + [("generic", q, None) for q in gen])

    from trigger_gate import TriggerGate
    gate = TriggerGate(s.llm, tok, s.pooler, "trigger_head_v3_gate.npz")

    res = {}
    for name, g in (("always", None), ("gated", gate)):
        print(f"\n=== arm={name} history={len(hist)} tokens ===", flush=True)
        res[name] = run_arm(name, g, questions, hist, s, tok)

    print("\n=== SUMMARY ===")
    for name, rows in res.items():
        ev = [r for r in rows if r["qtype"] == "evict"]
        wi = [r for r in rows if r["qtype"] == "window"]
        ge = [r for r in rows if r["qtype"] == "generic"]
        print(f"  {name:6s} evict acc {sum(r['ok'] for r in ev)}/{len(ev)} "
              f"fired {sum(r['fired'] for r in ev)}/{len(ev)} | "
              f"window acc {sum(r['ok'] for r in wi)}/{len(wi)} "
              f"fired {sum(r['fired'] for r in wi)}/{len(wi)} | "
              f"generic fired {sum(r['fired'] for r in ge)}/{len(ge)}")
    with open("gated_needle_results.json", "w") as f:
        json.dump(res, f, indent=1)
    print("NEEDLE_GATED_DONE")


if __name__ == "__main__":
    main()
