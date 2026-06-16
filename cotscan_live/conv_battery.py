import os, sys, time, types
BASE=os.path.join(os.path.dirname(os.path.abspath(__file__)), "faithful_run")
sys.path.insert(0, BASE); sys.path.insert(0, os.path.join(BASE,"runtime")); os.chdir(BASE)
import demo_chat
args = types.SimpleNamespace(no_web=True, resume=False, cap=900, seed=0, script=None)
s, mem = demo_chat.build_session(args)
LOG=[]
def log(x): print(x, flush=True); LOG.append(x); open("conv_battery.log","w").write("\n".join(LOG))
log("[battery UNIFIED v3 (FIX5 chitchat-frame)]")
script = [
  "Hey there!",                                   # greeting
  "I love hiking on weekends.",                   # declarative chitchat (refusal test)
  "I just adopted two kittens.",                  # declarative chitchat
  "I'm a bit stressed about a work deadline.",    # emotional chitchat
  "Any quick tips to relax?",                     # open advice question
  "What's the capital of France?",                # known fact
  "Who wrote Romeo and Juliet?",                  # known fact
  "What is 18 times 7?",                          # math standalone
  "Now add 10 to that.",                          # math follow-up (anaphora)
  "Remember my flight number is JL412.",          # save
  "My trip budget is 47000 dollars.",             # implicit fact statement
  "The weather has been lovely lately.",          # chitchat filler
  "I tried a great ramen place yesterday.",       # chitchat filler
  "What was my flight number again?",             # recall after distraction
  "And what's my budget?",                        # recall
  "Write me a short two-line poem about autumn.", # creative
  "Tell me a fun fact about the ocean.",          # open knowledge
]
for msg in script:
    t=time.time(); ans,src,ch=s.turn(msg, store="session")
    log(f"\nUSER> {msg}\nBOT> {ans}\n  [{time.time()-t:.0f}s src={src} chunks={len(ch)}]")
log("\n[DONE]")
