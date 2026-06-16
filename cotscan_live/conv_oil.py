import os, sys, time, types
BASE=os.path.join(os.path.dirname(os.path.abspath(__file__)), "faithful_run")
sys.path.insert(0, BASE); sys.path.insert(0, os.path.join(BASE,"runtime")); os.chdir(BASE)
import demo_chat, memory_core as mc
args = types.SimpleNamespace(no_web=False, resume=False, cap=900, seed=0, script=None)  # WEB ON
s, mem = demo_chat.build_session(args)
LOG=[]
def log(x): print(x, flush=True); LOG.append(x); open("conv_oil.log","w").write("\n".join(LOG))
log("[OIL diag: web trigger across phrasings, WEB ON]")
for msg in [
  "Tell me today's crude oil price.",        # EN imperative ("tell me" -> is_question True)
  "What is the current price of crude oil?",  # EN question
  "今日の原油価格を教えて",                    # JP imperative, no '?'
  "原油価格はいくら？",                        # JP question, full-width ？
]:
    log(f"\n--- is_question={mc._is_question(msg)} ---")
    t=time.time(); ans,src,ch=s.turn(msg, store="session")
    log(f"USER> {msg}\nBOT> {ans}\n  [{time.time()-t:.0f}s src={src} chunks={len(ch)}]")
    for j,c in enumerate(ch): log(f"    CHUNK[{j}]: {c[:160]}")
log("\n[DONE]")
