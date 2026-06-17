#!/bin/bash
# App-quality recall test on the faithful runtime with recall_lora_v2 merged in.
# Self-contained (watchdog self-destruct + HF heartbeat + self-upload + self-terminate).
#  - needle_recall_test.py (repo's Phase-A acceptance harness) BASE vs MERGED, conds sp,qk
#    -> does the recall-aware adapter improve real in-runtime block recall (via _gen_once)?
#  - conv_app.py full AppSession.turn() transcript (qualitative app quality, no-web).
# Requires env: HF_TOKEN, RUNPOD_KEY, plus optional CONDS.
exec > /workspace/boot.log 2>&1
set -x
export HF_HUB_ENABLE_HF_TRANSFER=0 PYTHONUNBUFFERED=1 SPCHAT_DEVICE=cuda
cd /workspace
PID="${RUNPOD_POD_ID:-unknown}"; MAXSEC="${MAXSEC:-5400}"; CONDS="${CONDS:-sp,qk}"
selfkill(){ curl -s -X DELETE -H "Authorization: Bearer $RUNPOD_KEY" "https://rest.runpod.io/v1/pods/$PID"; }
( sleep "$MAXSEC"; echo "WATCHDOG fired"; selfkill ) &

pip install -q "huggingface_hub>=0.25" 2>&1 | tail -1
cat > /workspace/mark.py <<'PY'
import sys,os,io
from huggingface_hub import HfApi
HfApi(token=os.environ.get("HF_TOKEN")).upload_file(
    path_or_fileobj=io.BytesIO(sys.argv[1].encode()),
    path_in_repo="apprecall_status.txt", repo_id="baya1116/hypernet-sp-distill")
print("marked:",sys.argv[1])
PY
mark(){ python /workspace/mark.py "$1 @ $(date -u +%H:%M:%S)" 2>&1 | tail -1; }
mark "BOOT_STARTED app-recall"

pip install -q "transformers>=4.46,<4.54" "peft>=0.11" "joblib" "scikit-learn" "numpy<2" 2>&1 | tail -1
# reconstruct the faithful runtime from public HF (hypernet_sp/* flat, runtime/*, evals/*, weights)
python - <<'PY'
from huggingface_hub import hf_hub_download, HfApi
import shutil, os, fnmatch
R="baya1116/hypernet-sp-distill"
os.makedirs("runtime",exist_ok=True); os.makedirs("evals",exist_ok=True); os.makedirs("fft_out",exist_ok=True)
os.makedirs("recall_lora_v2",exist_ok=True)
files=[f.rfilename for f in HfApi().model_info(R).siblings]
def g(p,dst): shutil.copy(hf_hub_download(R,p),dst)
for p in [f for f in files if fnmatch.fnmatch(f,"hypernet_sp/*.py")]: g(p,os.path.basename(p))
for p in ["runtime/rag.py","runtime/web_search.py","runtime/cache.py","runtime/refusal.py","runtime/retrieval_pipeline.py"]:
    try: g(p,os.path.join("runtime",os.path.basename(p)))
    except Exception as e: print("skip",p,e)
for p in ["evals/intent_clf.joblib","evals/specificity_clf.joblib"]: g(p,os.path.join("evals",os.path.basename(p)))
g("build_fft_hf.py","build_fft_hf.py")
g("fft_out/student.pt","fft_out/student.pt"); g("fft_out/pooler.pt","fft_out/pooler.pt")
for p in ["recall_lora_v2/adapter_config.json","recall_lora_v2/adapter_model.safetensors"]:
    g(p,os.path.join("recall_lora_v2",os.path.basename(p)))
print("reconstructed", len([f for f in files if f.endswith('.py')]),"py files")
PY
python build_fft_hf.py 2>&1 | tail -2
ls -d fft_hf >/dev/null 2>&1 && mark "FFT_HF_OK" || mark "FFT_HF_MISSING"

# tag-wrap the runtime's recall injection to match training ((Recalled from earlier: "..."))
python - <<'PY'
import re
p="app_session_torch.py"; s=open(p).read()
new='if rid:\n                    rid = tok.encode(\'(Recalled from earlier: "\', add_special_tokens=False) + rid + tok.encode(\'")\', add_special_tokens=False)\n                    rec_emb = self._emb(rid)'
s2,n=re.subn(r"if rid:\s*\n\s*rec_emb = self\._emb\(rid\)", new, s)
open(p,"w").write(s2); print("tagwrap replacements:",n)
PY
# best-effort apply the full faithful patch (router removal etc.) for the conversation turn()
curl -fsSL "https://raw.githubusercontent.com/bayamax/dual-masked-bert-jp/claude/hypernet-sp-distill-d3pyik/cotscan_live/faithful_patches/app_session_torch.patch" -o app.patch || true
patch -s --forward app_session_torch.py app.patch && echo "FULL_PATCH_OK" || echo "FULL_PATCH_SKIP (conv uses canonical+tagwrap)"

B="https://raw.githubusercontent.com/bayamax/dual-masked-bert-jp/claude/hypernet-sp-distill-d3pyik/cotscan_live/recalltrain"
curl -fsSL "$B/conv_app.py" -o conv_app.py

# 1) needle acceptance: BASE
mark "NEEDLE_BASE conds=$CONDS"
python needle_recall_test.py --conds "$CONDS" 2>&1 | tee needle_base.log
cp needle_results.json needle_base.json 2>/dev/null || true

# 2) merge recall_lora_v2 into fft_hf
python - <<'PY'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
try: m=AutoModelForCausalLM.from_pretrained("fft_hf",torch_dtype=torch.float32)
except TypeError: m=AutoModelForCausalLM.from_pretrained("fft_hf",dtype=torch.float32)
m=PeftModel.from_pretrained(m,"recall_lora_v2").merge_and_unload()
m.save_pretrained("fft_hf_merged"); AutoTokenizer.from_pretrained("fft_hf").save_pretrained("fft_hf_merged")
print("merged ok")
PY
rm -rf fft_hf && mv fft_hf_merged fft_hf

# 3) needle acceptance: MERGED (recall_lora_v2)
mark "NEEDLE_MERGED conds=$CONDS"
python needle_recall_test.py --conds "$CONDS" 2>&1 | tee needle_merged.log
cp needle_results.json needle_merged.json 2>/dev/null || true

# 4) qualitative full-turn() conversation transcript (merged, qk recall, no-web)
mark "CONV_APP"
python conv_app.py 2>&1 | tee conv_app_run.log || echo "conv_app failed (non-blocking)"

# upload everything to HF then self-terminate
python - <<'PY'
import os
from huggingface_hub import HfApi
api=HfApi(token=os.environ.get("HF_TOKEN")); R="baya1116/hypernet-sp-distill"
def up(l,r):
    if os.path.exists(l): api.upload_file(path_or_fileobj=l,path_in_repo=r,repo_id=R); print("up",r)
for l,r in [("needle_base.json","apprecall_needle_base.json"),("needle_merged.json","apprecall_needle_merged.json"),
            ("needle_base.log","apprecall_needle_base.log"),("needle_merged.log","apprecall_needle_merged.log"),
            ("conv_app.log","apprecall_conv.log"),("conv_app_run.log","apprecall_conv_run.log"),("boot.log","apprecall_boot.log")]:
    up(l,r)
print("HF_UPLOAD_DONE")
PY
mark "UPLOADED self-terminate"
selfkill
