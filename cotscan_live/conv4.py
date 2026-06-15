import os, sys, time, types
BASE=os.path.join(os.path.dirname(os.path.abspath(__file__)), "faithful_run")
sys.path.insert(0, BASE); sys.path.insert(0, os.path.join(BASE,"runtime")); os.chdir(BASE)
import demo_chat
args = types.SimpleNamespace(no_web=True, resume=False, cap=900, seed=0, script=None)
s, mem = demo_chat.build_session(args)
LOG=[]
def log(x): print(x, flush=True); LOG.append(x); open("conv4.log","w").write("\n".join(LOG))
log("[built persona-fix]")
for msg in [
  "Hi! I just adopted two kittens.",
  "What would be good names for them?",
  "Tell me a fun fact about cats.",
]:
    t=time.time(); ans,src,ch=s.turn(msg, store="session")
    log(f"\nUSER> {msg}\nBOT> {ans}\n  [{time.time()-t:.0f}s src={src} chunks={len(ch)}]")
log("\n[DONE]")
