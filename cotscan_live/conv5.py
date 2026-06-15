import os, sys, time, types
BASE=os.path.join(os.path.dirname(os.path.abspath(__file__)), "faithful_run")
sys.path.insert(0, BASE); sys.path.insert(0, os.path.join(BASE,"runtime")); os.chdir(BASE)
import demo_chat
args = types.SimpleNamespace(no_web=True, resume=False, cap=900, seed=0, script=None)
s, mem = demo_chat.build_session(args)
LOG=[]
def log(x): print(x, flush=True); LOG.append(x); open("conv5.log","w").write("\n".join(LOG))
log("[built UNIFIED no-router]")
for msg in [
  "Hey, I just adopted two kittens!",                  # chitchat (no refusal?)
  "Remember my flight number is JL412.",               # explicit save -> ack
  "My budget for the trip is 47000 dollars.",          # implicit fact (now generates?)
  "The weather has been lovely lately.",               # casual chitchat
  "What's the capital of France?",                     # KNOWN -> should answer Paris offline (#3)
  "What is 18 times 7?",                               # math
  "What was my flight number again?",                  # recall
]:
    t=time.time(); ans,src,ch=s.turn(msg, store="session")
    log(f"\nUSER> {msg}\nBOT> {ans}\n  [{time.time()-t:.0f}s src={src} chunks={len(ch)}]")
log("\n[DONE]")
