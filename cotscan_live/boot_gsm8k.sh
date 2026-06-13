#!/bin/bash
set +e
exec > /workspace/boot.log 2>&1
echo "BOOT_START $(date -u)"
export HF_HUB_ENABLE_HF_TRANSFER=0
RID="baya1116/hypernet-sp-distill"
OUT="trigger_experiment/results_gsm8k"

pip install -q -U "huggingface_hub>=0.25" "transformers>=4.46,<4.54" "datasets>=2.18" requests 2>&1 | tail -3
echo "PIP_DONE $(date -u)"
mkdir -p /work/hypernet_sp && cd /work/hypernet_sp

python - <<'PY'
import os, shutil
from huggingface_hub import hf_hub_download
R="baya1116/hypernet-sp-distill"; tok=os.environ["HF_TOKEN"]
for p in ["hypernet_sp/block_recall.py","hypernet_sp/attn_export3_torch.py",
          "hypernet_sp/attn_scenarios.py","build_fft_hf.py",
          "fft_out/student.pt","fft_out/pooler.pt"]:
    f=hf_hub_download(R,p,token=tok)
    dst=p if p.startswith("fft_out/") else os.path.basename(p)
    os.makedirs(os.path.dirname(dst) or ".",exist_ok=True); shutil.copy(f,dst)
    print("got",dst,os.path.getsize(dst))
print("DOWNLOAD_DONE")
PY
echo "DL_DONE $(date -u)"

echo "$GSM8K_PY_B64" | base64 -d > cotscan_gsm8k.py
wc -l cotscan_gsm8k.py
echo "BUILD_FFT_START $(date -u)"; python build_fft_hf.py; echo "BUILD_FFT_DONE $(date -u)"

python - <<'PY' &
import time, os
from huggingface_hub import HfApi
api=HfApi(token=os.environ["HF_TOKEN"]); R="baya1116/hypernet-sp-distill"
OUT="trigger_experiment/results_gsm8k"
while True:
    for fn in ["live.log","results.json"]:
        p=f"results_gsm8k/{fn}"
        if os.path.exists(p):
            try: api.upload_file(path_or_fileobj=p, path_in_repo=f"{OUT}/{fn}", repo_id=R)
            except Exception as e: print("up_err",e,flush=True)
    if os.path.exists("results_gsm8k/.done"): break
    time.sleep(20)
print("UPLOADER_EXIT",flush=True)
PY

echo "RUN_START $(date -u)"
GSM8K_N=${GSM8K_N:-25} OUT_DIR=results_gsm8k python cotscan_gsm8k.py
echo "RUN_DONE $(date -u) rc=$?"
touch results_gsm8k/.done

python - <<'PY'
import os
from huggingface_hub import HfApi
api=HfApi(token=os.environ["HF_TOKEN"]); R="baya1116/hypernet-sp-distill"
OUT="trigger_experiment/results_gsm8k"
for p,name in [("results_gsm8k/results.json","results.json"),
               ("results_gsm8k/live.log","live.log"),("/workspace/boot.log","boot.log")]:
    if os.path.exists(p):
        try: api.upload_file(path_or_fileobj=p, path_in_repo=f"{OUT}/{name}", repo_id=R); print("flushed",name)
        except Exception as e: print("flush_err",name,e)
PY
echo "FLUSH_DONE $(date -u)"

python - <<'PY'
import os, requests
pid=os.environ.get("RUNPOD_POD_ID"); key=os.environ.get("RUNPOD_API_KEY")
if pid and key:
    requests.post("https://api.runpod.io/graphql",
        headers={"Authorization":f"Bearer {key}","Content-Type":"application/json"},
        json={"query":'mutation{podTerminate(input:{podId:"%s"})}'%pid}, timeout=30)
    print("TERMINATE sent", pid)
PY
echo "BOOT_END $(date -u)"
