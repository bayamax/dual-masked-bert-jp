import os, sys, time, types
BASE=os.path.join(os.path.dirname(os.path.abspath(__file__)), "faithful_run")
sys.path.insert(0, BASE); sys.path.insert(0, os.path.join(BASE,"runtime")); os.chdir(BASE)
for f in ("demo_memory.jsonl","demo_state.json"):
    try: os.remove(f)
    except OSError: pass
import demo_chat, memory_core as mc
args = types.SimpleNamespace(no_web=False, resume=False, cap=900, seed=0, script=None)  # WEB ON
s, mem = demo_chat.build_session(args)
LOG=[]
def log(x): print(x, flush=True); LOG.append(x); open("conv_repro.log","w").write("\n".join(LOG))

def reset():
    s.gen, s.kept, s.absorbed, s.evictions = [], [], 0, 0
    s.mem.session=[]; s.mem.pins=[]; s.mem.persistent=[]

def run(title, script):
    reset(); log(f"\n======== {title} ========")
    for msg in script:
        t=time.time(); ans,src,ch=s.turn(msg, store="session")
        log(f"\nUSER> {msg}   (is_question={mc._is_question(msg)})\nBOT> {ans}\n  [{time.time()-t:.0f}s src={src} chunks={len(ch)}]")
        for j,c in enumerate(ch): log(f"    CHUNK[{j}]: {c[:140]}")

# Reproduce the screenshot flow: greeting WITH A NAME, then the oil query, then a rephrase.
run("A: English (post-translation form)", [
    "Hi, I'm Dai.",
    "Tell me today's crude oil price.",
    "I'd like you to tell me today's crude oil price.",
])
run("B: Japanese (untranslated, as a control)", [
    "こんにちは、Daiです。",
    "今日の原油価格を教えてください",
    "今日の原油価格を教えて欲しいのです",
])
log("\n[DONE]")
