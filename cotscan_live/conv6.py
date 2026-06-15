import os, sys, time, types
BASE=os.path.join(os.path.dirname(os.path.abspath(__file__)), "faithful_run")
sys.path.insert(0, BASE); sys.path.insert(0, os.path.join(BASE,"runtime")); os.chdir(BASE)
import demo_chat
args = types.SimpleNamespace(no_web=True, resume=False, cap=900, seed=0, script=None)
s, mem = demo_chat.build_session(args)
LOG=[]
def log(x): print(x, flush=True); LOG.append(x); open("conv6.log","w").write("\n".join(LOG))
log("[built UNIFIED v2]")
for msg in [
  "What is 18 times 7?",                  # standalone math -> 126, no irrelevant context (#9)
  "Now add 10 to that.",                  # follow-up math back-ref -> 136 ?
  "What's the capital of Japan?",         # known -> Tokyo offline
  "Remember my locker code is B12.",      # save
  "I love hiking on weekends.",           # statement chitchat
  "What's my locker code?",               # recall -> B12
]:
    t=time.time(); ans,src,ch=s.turn(msg, store="session")
    log(f"\nUSER> {msg}\nBOT> {ans}\n  [{time.time()-t:.0f}s src={src} chunks={len(ch)}]")
log("\n[DONE]")
