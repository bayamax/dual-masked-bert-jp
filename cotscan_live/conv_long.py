import os, sys, time, types
BASE=os.path.join(os.path.dirname(os.path.abspath(__file__)), "faithful_run")
sys.path.insert(0, BASE); sys.path.insert(0, os.path.join(BASE,"runtime")); os.chdir(BASE)
for f in ("demo_memory.jsonl","demo_state.json"):
    try: os.remove(f)
    except OSError: pass
import demo_chat
args = types.SimpleNamespace(no_web=True, resume=False, cap=900, seed=0, script=None)  # web OFF: isolate the multi-turn degradation
s, mem = demo_chat.build_session(args)
LOG=[]
def log(x): print(x, flush=True); LOG.append(x); open("conv_long.log","w").write("\n".join(LOG))
log("[LONG multi-turn degradation diag, web OFF]")
script = [
  "Hey, how's it going?",
  "I'm planning a weekend trip to Kyoto.",
  "I've been there once before and loved the old temples.",
  "What are some must-see spots there?",
  "Nice. I'm also really into photography.",
  "Any tips for shooting in low light?",
  "By the way, my camera is a Sony A7.",
  "What lens would you recommend for street photography?",
  "Remind me what camera I have?",
  "What was the first city I said I wanted to visit?",
  "I might extend the trip by two days.",
  "Can you summarize my trip plan so far?",
]
for i,msg in enumerate(script):
    t=time.time(); ans,src,ch=s.turn(msg, store="session")
    log(f"\n[turn {i+1}] USER> {msg}")
    log(f"BOT> {ans}")
    log(f"  [{time.time()-t:.0f}s src={src} chunks={len(ch)} pins={len(s.mem.pins)} kept={len(s.kept)} evict={s.evictions} session={len(s.mem.session)}]")
    for j,c in enumerate(ch): log(f"    CHUNK[{j}]: {c[:120]}")
    if s.mem.pins: log(f"    PINS: {[p[:50] for p in s.mem.pins]}")
log("\n[DONE]")
