"""demo_b_composite.py — composite multi-turn chat test: B-method session memory
(BGE + learned indexer ensemble) + external knowledge (real web) on the full turn()
pipeline (routing, tiered memory, honesty gates, calculator, stitching).

Run from /tmp/hf-sp:  PYTHONPATH=runtime:hypernet_sp python3 hypernet_sp/demo_b_composite.py
"""
import os, re, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.getcwd(), "runtime"))

FILLER_USERS = [
    "The weather has been all over the place this week, sunny mornings and then sudden storms in the afternoon. It made the commute messy twice and I had to dry my jacket on the office chair. Do you have any tricks for planning around weather like this?",
    "I tried that new ramen place near the station yesterday. The broth was a thick pork style with charred garlic oil, and the noodles were firmer than I expected. The line took twenty minutes but honestly it might be worth it. What makes a great bowl in your view?",
    "I started reading a novel about Antarctic explorers and it is gripping so far. The chapter about rationing fuel during a whiteout had me holding my breath. The author apparently used real expedition diaries as source material. Do stories like that appeal to you?",
    "My neighbor's dog learned to open the gate somehow, clever thing. It lifts the latch with its nose and trots straight to the park by itself. The neighbor added a carabiner but the dog is already studying it. Any ideas for dog-proofing a latch humanely?",
    "We are thinking about repainting the kitchen a lighter color, maybe a soft sage or a warm off-white. The cabinets are old oak so we want something that will not clash. The room only gets morning light. Which direction would you lean and why?",
    "The train was delayed twenty minutes this morning because of a signal inspection, and the platform got so crowded I let two trains pass before boarding. I used the time to clear my inbox at least. How do you usually make delays feel less wasted?",
    "I have been trying to drink more water instead of coffee lately, keeping a big bottle on the desk where I can see it. The afternoon headaches are already milder, though I miss the ritual of the third cup. Any advice for keeping the habit going?",
    "The garden tomatoes are finally turning red after weeks of stubborn green. We staked them late so a few vines sprawl along the ground, but the flavor of the first ripe one was incredible. Should I prune the lower leaves this late in the season?",
    "I watched a documentary about deep sea creatures last night and the anglerfish segment was unreal. There was also a translucent octopus that seemed to glow from inside. The pressure down there crushes submarines yet life thrives. What part of the ocean fascinates you?",
    "My phone battery barely lasts the afternoon now, even after I dimmed the screen and disabled background refresh for most apps. It is about three years old so maybe the cells are just tired. Is a battery swap worth it or should I hold out for a new phone?",
    "We played board games with friends over the weekend, a long strategy one about trading routes and a fast bluffing card game after midnight. I lost both but the table talk was the real prize. Do you think losing gracefully is a learnable skill?",
    "I am learning to make sourdough bread and the starter is fussy, bubbling happily one day and sulking flat the next. I feed it on a schedule and keep it warm near the oven. The last loaf had a great crust but dense crumb. What should I adjust first?",
]

def checks():
    T = []
    T.append(("Hello!", lambda a: not re.search(r"\bsolve|equation|=\s*\d", a or ""), "greet-no-task"))
    T.append(("I just got back from Takayama, the old town was beautiful.", lambda a: bool(a), "chitchat"))
    T.append(("Please remember: my project codename is Aurora-7.", lambda a: "saved" in (a or "").lower() or "got it" in (a or "").lower(), "persist-ack"))
    T.append(("By the way, the rental car plate ended in 4471, kind of memorable.", lambda a: bool(a), "casual-fact-1"))
    T.append(("My hotel room was 318, up on the third floor.", lambda a: bool(a), "casual-fact-2"))
    for u in FILLER_USERS:
        T.append((u, lambda a: bool(a), "filler"))
    T.append(("What did the rental car plate end in?", lambda a: "4471" in (a or ""), "deep-recall-1"))
    T.append(("And which room was I in at the hotel?", lambda a: "318" in (a or ""), "deep-recall-2"))
    T.append(("How tall is Mount Fuji?", lambda a: re.search(r"3,?77[0-9]", (a or "").replace(",", "")), "web-lookup"))
    T.append(("Can you verify that number?", lambda a: re.search(r"3,?77[0-9]", (a or "").replace(",", "")) or "fuji" in (a or "").lower(), "verify-anaphora"))
    T.append(("What's my project codename again?", lambda a: "aurora" in (a or "").lower(), "L2-recall"))
    T.append(("A bento costs $9 and I buy 4 for the team. What's the total?", lambda a: "36" in (a or ""), "math"))
    T.append(("Earlier we looked something up on the web - what was it about?", lambda a: "fuji" in (a or "").lower() or "mountain" in (a or "").lower(), "weblog-recall"))
    T.append(("Write a two-line poem about my trip.", lambda a: re.search(r"[A-Za-z]{3,}", a or "") and "4471" not in (a or ""), "poem-no-echo"))
    T.append(("Thanks, this was fun!", lambda a: bool(a) and not re.search(r"^\s*\d+\.?\s*$", a or ""), "close"))
    return T


def main():
    import torch, joblib
    torch.set_num_threads(os.cpu_count())
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from rag import BGERetriever
    import memory_core as mc
    from app_session_torch import AppSession
    from attn_export3_torch import load_pooler
    from ensemble_archive import EnsembleArchive
    from demo_chat import RealWeb
    print("DEMO_B_COMPOSITE_V1 loading ...", flush=True)
    tok = AutoTokenizer.from_pretrained("fft_hf")
    llm = AutoModelForCausalLM.from_pretrained("fft_hf", dtype=torch.float32).eval()
    pooler, bge = load_pooler(), BGERetriever()
    for f in ("/tmp/demo_b_mem.jsonl",):
        if os.path.exists(f):
            os.remove(f)
    mem = mc.TieredMemory("/tmp/demo_b_mem.jsonl", bge=bge)
    mem.session, mem.persistent, mem.pins = [], [], []
    web = RealWeb()
    archive = EnsembleArchive(llm, tok, bge)
    s = AppSession(llm, tok, pooler, bge, joblib.load("evals/intent_clf.joblib"),
                   joblib.load("evals/specificity_clf.joblib"), mem, web=web,
                   cap=420, seed=0, archive=archive)
    results = []
    for u, chk, name in checks():
        t0 = time.time()
        try:
            ans, src, chunks = s.turn(u, store="session")
        except Exception as e:
            ans, src, chunks = f"[ERROR {e}]", None, []
        ok = bool(chk(ans))
        results.append((name, ok))
        print(f"[{'PASS' if ok else 'FAIL'}] {name} ({time.time()-t0:.0f}s | "
              f"stream={len(s.gen)} blocks={len(archive.blocks)} src={src})\n"
              f"  U: {u}\n  A: {(ans or '')[:180]}", flush=True)
    named = [(n, o) for n, o in results if n != "filler"]
    print(f"\nDEMO_B_SCORE {sum(o for _, o in named)}/{len(named)} "
          f"(fillers responded: {sum(o for n, o in results if n == 'filler')}/12)", flush=True)
    print("DEMO_B_DONE", flush=True)


if __name__ == "__main__":
    main()
