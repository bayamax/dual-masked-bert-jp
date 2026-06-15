"""faithful_setup.py — reconstruct the faithful on-device runtime locally (CPU) from the
PUBLIC HF repo baya1116/hypernet-sp-distill, so we can converse with the canonical
AppSession.turn() and iterate. Run from cotscan_live/:  python3 faithful_setup.py
Produces faithful_run/ with: fft_hf/ (built from fft_out/student.pt), fft_out/pooler.pt,
the hypernet_sp/* runtime modules, runtime/{rag,web_search,...}.py, evals/*.joblib.
Then drive it with smoke.py / conv1.py (copied alongside).
"""
import os, shutil, fnmatch
from huggingface_hub import hf_hub_download, HfApi
R = "baya1116/hypernet-sp-distill"
RUN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faithful_run")
os.makedirs(os.path.join(RUN, "runtime"), exist_ok=True)
os.makedirs(os.path.join(RUN, "evals"), exist_ok=True)
os.makedirs(os.path.join(RUN, "fft_out"), exist_ok=True)
files = [f.rfilename for f in HfApi().model_info(R).siblings]

def grab(p, dst):
    shutil.copy(hf_hub_download(R, p), dst)

for p in [f for f in files if fnmatch.fnmatch(f, "hypernet_sp/*.py")]:
    grab(p, os.path.join(RUN, os.path.basename(p)))
for p in ["runtime/rag.py", "runtime/web_search.py", "runtime/cache.py",
          "runtime/refusal.py", "runtime/retrieval_pipeline.py"]:
    grab(p, os.path.join(RUN, "runtime", os.path.basename(p)))
for p in ["evals/intent_clf.joblib", "evals/specificity_clf.joblib"]:
    grab(p, os.path.join(RUN, "evals", os.path.basename(p)))
grab("build_fft_hf.py", os.path.join(RUN, "build_fft_hf.py"))
grab("fft_out/student.pt", os.path.join(RUN, "fft_out", "student.pt"))
grab("fft_out/pooler.pt", os.path.join(RUN, "fft_out", "pooler.pt"))
# transformers>=4.53 renamed dtype kw; patch the canonical loader call
dc = os.path.join(RUN, "demo_chat.py"); s = open(dc).read()
open(dc, "w").write(s.replace(", dtype=torch.float32", ", torch_dtype=torch.float32"))
print("faithful_run ready; cd faithful_run && python3 build_fft_hf.py  then  python3 ../smoke.py")
