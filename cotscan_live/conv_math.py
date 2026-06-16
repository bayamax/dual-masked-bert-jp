import os, sys, time, types
BASE=os.path.join(os.path.dirname(os.path.abspath(__file__)), "faithful_run")
sys.path.insert(0, BASE); sys.path.insert(0, os.path.join(BASE,"runtime")); os.chdir(BASE)
import demo_chat
args = types.SimpleNamespace(no_web=True, resume=False, cap=900, seed=0, script=None)
s, mem = demo_chat.build_session(args)
LOG=[]
def log(x): print(x, flush=True); LOG.append(x); open("conv_math.log","w").write("\n".join(LOG))
log("[MATH-DISCUSSION battery: coherent word-problem context]")
# A coherent, math-centric conversation (the realistic scenario), with a little chitchat woven in,
# plus the SAME arithmetic asked twice (turns 2 and 11) to gauge stochastic flakiness in-context.
script = [
  "I'm organizing a small party and need help with the numbers.",        # set context
  "There will be 18 guests and each eats about 7 snacks. How many snacks total?",  # ->126
  "Snacks come in packs of 10. How many packs do I need?",               # ->13
  "Each pack costs 3 dollars. What's the total snack cost?",              # ->39
  "I also want 2 drinks per guest. How many drinks is that?",            # ->36
  "What's the combined number of snacks and drinks?",                    # 126+36 ->162
  "Honestly I'm excited, it's been a while since I hosted anything.",     # chitchat woven in
  "Right, back to numbers — if 4 guests cancel, how many snacks now?",   # 14*7 ->98
  "And how many packs for that?",                                        # ->10
  "Thanks! What was the ORIGINAL total number of snacks again?",         # recall ->126
  "Just to double check: what is 18 times 7?",                           # repeat arithmetic ->126
  "Great, that matches.",                                                # chitchat close
]
for msg in script:
    t=time.time(); ans,src,ch=s.turn(msg, store="session")
    log(f"\nUSER> {msg}\nBOT> {ans}\n  [{time.time()-t:.0f}s src={src} chunks={len(ch)}]")
    for j,c in enumerate(ch):
        log(f"    CHUNK[{j}]: {c[:160]}")
log("\n[DONE]")
