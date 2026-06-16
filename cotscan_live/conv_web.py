import os, sys, time, types
BASE=os.path.join(os.path.dirname(os.path.abspath(__file__)), "faithful_run")
sys.path.insert(0, BASE); sys.path.insert(0, os.path.join(BASE,"runtime")); os.chdir(BASE)
import demo_chat
args = types.SimpleNamespace(no_web=False, resume=False, cap=900, seed=0, script=None)  # WEB ON
s, mem = demo_chat.build_session(args)
LOG=[]
def log(x): print(x, flush=True); LOG.append(x); open("conv_web.log","w").write("\n".join(LOG))
log("[COMPOSITE recall x web battery, WEB ON]")
# interleave SAVE/RECALL (personal memory) with WEB lookups -> test coexistence in one convo
script = [
  ("Remember my project codename is Bluefin.",        "save"),
  ("What is the capital of Australia?",                "web"),
  ("What's my project codename?",                      "recall->Bluefin"),
  ("Who wrote the novel Pride and Prejudice?",         "web/known"),
  ("Remember my locker code is C12.",                  "save"),
  ("What is the population of Canberra?",              "web"),
  ("What was my locker code?",                         "recall->C12"),
  ("What is the chemical symbol for gold?",            "web/known"),
  ("Remind me both things I asked you to remember.",   "recall->Bluefin+C12"),
]
for msg, expect in script:
    t=time.time(); ans,src,ch=s.turn(msg, store="session")
    log(f"\nUSER> {msg}   (expect: {expect})\nBOT> {ans}\n  [{time.time()-t:.0f}s src={src} chunks={len(ch)}]")
    for j,c in enumerate(ch): log(f"    CHUNK[{j}]: {c[:140]}")
log("\n[DONE]")
