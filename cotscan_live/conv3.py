import os, sys, time, types
BASE=os.path.join(os.path.dirname(os.path.abspath(__file__)), "faithful_run")
sys.path.insert(0, BASE); sys.path.insert(0, os.path.join(BASE,"runtime")); os.chdir(BASE)
import demo_chat
args = types.SimpleNamespace(no_web=True, resume=False, cap=900, seed=0, script=None)
s, mem = demo_chat.build_session(args)
LOG=[]
def log(x): print(x, flush=True); LOG.append(x); open("conv3.log","w").write("\n".join(LOG))
log("[built]")
script = [
  "Hi! I just adopted two kittens.",                       # chitchat
  "What would be good names for them?",                    # anaphora 'them' -> kittens, chitchat
  "What is 18 times 7?",                                   # math -> check no \boxed leak
  "Now add 10 to that.",                                   # compute follow-up uses prior result
  "By the way, remember my parking spot is B12.",          # fact save
  "I'm feeling a bit stressed about a deadline this week.", # emotional chitchat
  "Any quick tips to calm down?",                          # advice chitchat
  "What was my parking spot again?",                       # recall
]
for msg in script:
    t=time.time(); ans,src,ch=s.turn(msg, store="session")
    log(f"\nUSER> {msg}\nBOT> {ans}\n  [{time.time()-t:.0f}s src={src} chunks={len(ch)} kept={len(s.kept)} evict={s.evictions}]")
log("\n[DONE]")
