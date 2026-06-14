#!/bin/bash
set +e
exec > /workspace/boot.log 2>&1
echo "BOOT_START $(date -u)"
export HF_HUB_ENABLE_HF_TRANSFER=0
pip install -q -U "huggingface_hub>=0.25" "transformers>=4.46,<4.54" "datasets>=2.18" requests 2>&1 | tail -2
echo "PIP_DONE $(date -u)"
mkdir -p /work && cd /work && mkdir -p results_gsm_sp512 results_gsm_sp1024
python - <<'PY'
import os, shutil
from huggingface_hub import hf_hub_download
R="baya1116/hypernet-sp-distill"; tok=os.environ["HF_TOKEN"]
for p in ["hypernet_sp/attn_export3_torch.py","hypernet_sp/attn_scenarios.py",
          "build_fft_hf.py","fft_out/student.pt","fft_out/pooler.pt"]:
    f=hf_hub_download(R,p,token=tok); dst=p if p.startswith("fft_out/") else os.path.basename(p)
    os.makedirs(os.path.dirname(dst) or ".",exist_ok=True); shutil.copy(f,dst)
print("HF_DONE")
PY
echo "$SP_PY_B64" | base64 -d > gsm8k_sp.py
echo "BUILD_FFT $(date -u)"; python build_fft_hf.py 2>&1 | tail -1; echo "BUILD_FFT_DONE $(date -u)"
python - <<'PY' &
import time, os
from huggingface_hub import HfApi
api=HfApi(token=os.environ["HF_TOKEN"]); R="baya1116/hypernet-sp-distill"
dirs={"results_gsm_sp512":"trigger_experiment/results_gsm_sp512","results_gsm_sp1024":"trigger_experiment/results_gsm_sp1024"}
while True:
    for d,rep in dirs.items():
        for fn in ["live.log","results.json"]:
            p=f"{d}/{fn}"
            if os.path.exists(p):
                try: api.upload_file(path_or_fileobj=p,path_in_repo=f"{rep}/{fn}",repo_id=R)
                except Exception as e: print("up",e,flush=True)
    if os.path.exists("DONE"): break
    time.sleep(20)
PY
echo "RUN512 $(date -u)"; RW=512 GSM_N=${GSM_N:-30} OUT_DIR=results_gsm_sp512 python -u gsm8k_sp.py
echo "RUN1024 $(date -u)"; RW=1024 GSM_N=${GSM_N:-30} OUT_DIR=results_gsm_sp1024 python -u gsm8k_sp.py
echo "RUNS_DONE $(date -u)"; touch DONE; sleep 22
python - <<'PY'
import os
from huggingface_hub import HfApi
api=HfApi(token=os.environ["HF_TOKEN"]); R="baya1116/hypernet-sp-distill"
for d,rep in [("results_gsm_sp512","trigger_experiment/results_gsm_sp512"),("results_gsm_sp1024","trigger_experiment/results_gsm_sp1024")]:
    for fn in ["results.json","live.log"]:
        p=f"{d}/{fn}"
        if os.path.exists(p):
            try: api.upload_file(path_or_fileobj=p,path_in_repo=f"{rep}/{fn}",repo_id=R)
            except Exception as e: print("ferr",e)
print("FLUSH")
PY
python - <<'PY'
import os, requests
pid=os.environ.get("RUNPOD_POD_ID");key=os.environ.get("RUNPOD_API_KEY")
if pid and key: requests.post("https://api.runpod.io/graphql",headers={"Authorization":f"Bearer {key}","Content-Type":"application/json"},json={"query":'mutation{podTerminate(input:{podId:"%s"})}'%pid},timeout=30); print("TERM")
PY
echo "BOOT_END $(date -u)"
