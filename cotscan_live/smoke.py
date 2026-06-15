import os, sys, time, types
BASE=os.path.join(os.path.dirname(os.path.abspath(__file__)), "faithful_run")
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "runtime"))
os.chdir(BASE)
import demo_chat
args = types.SimpleNamespace(no_web=True, resume=False, cap=900, seed=0, script=None)
t0=time.time(); s, mem = demo_chat.build_session(args); print(f"[built in {time.time()-t0:.0f}s]", flush=True)
for msg in ["Hi there!", "My name is Alex and my favorite color is teal."]:
    t=time.time(); ans,src,ch = s.turn(msg, store="session")
    print(f"\nUSER> {msg}\nBOT> {ans}\n  [{time.time()-t:.0f}s src={src} chunks={len(ch)} kept={len(s.kept)}]", flush=True)
