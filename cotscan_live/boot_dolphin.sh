#!/bin/bash
set +e
exec > /workspace/boot.log 2>&1
echo "BOOT_START $(date -u)"
export HF_HUB_ENABLE_HF_TRANSFER=0
GH="https://raw.githubusercontent.com/bayamax/dual-masked-bert-jp/e664ec9bb30b5a9a536439f0f87f498d887e2a31/trigger_experiment"
O="trigger_experiment/results_dolphin"

pip install -q -U "huggingface_hub>=0.25" "transformers>=4.46,<4.54" "datasets>=2.18" "scikit-learn>=1.3" requests 2>&1 | tail -2
echo "PIP_DONE $(date -u)"
mkdir -p /work && cd /work && mkdir -p results_dolphin results_dolphin_smoke

python - <<'PY'
import os, shutil
from huggingface_hub import hf_hub_download
R="baya1116/hypernet-sp-distill"; tok=os.environ["HF_TOKEN"]
for p in ["hypernet_sp/attn_export3_torch.py","hypernet_sp/attn_scenarios.py",
          "hypernet_sp/block_recall.py","build_fft_hf.py","fft_out/student.pt","fft_out/pooler.pt"]:
    f=hf_hub_download(R,p,token=tok); dst=p if p.startswith("fft_out/") else os.path.basename(p)
    os.makedirs(os.path.dirname(dst) or ".",exist_ok=True); shutil.copy(f,dst)
print("HF_DONE")
PY
python - <<PY
import urllib.request
GH="$GH"
open("cot_recall_eval.py","wb").write(urllib.request.urlopen(f"{GH}/cot_recall_eval.py",timeout=60).read())
print("got cot_recall_eval.py")
PY
sed -i 's/, dtype=torch\.float32/, torch_dtype=torch.float32/g' cot_recall_eval.py 2>/dev/null
echo "$DOLPHIN_PY_B64" | base64 -d > dolphin_scan.py
echo "BUILD_FFT $(date -u)"; python build_fft_hf.py 2>&1 | tail -1; echo "BUILD_FFT_DONE $(date -u)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# uploader: stream both smoke and full out dirs
python - <<'PY' &
import time, os
from huggingface_hub import HfApi
api=HfApi(token=os.environ["HF_TOKEN"]); R="baya1116/hypernet-sp-distill"
for _ in range(100000):
    for d,rep in [("results_dolphin","trigger_experiment/results_dolphin"),
                  ("results_dolphin_smoke","trigger_experiment/results_dolphin_smoke")]:
        if os.path.isdir(d):
            for fn in os.listdir(d):
                try: api.upload_file(path_or_fileobj=os.path.join(d,fn),path_in_repo=f"{rep}/{fn}",repo_id=R)
                except Exception as e: print("up_err",fn,e,flush=True)
    if os.path.exists("DONE"): break
    time.sleep(20)
PY

echo "SMOKE_START $(date -u)"
DOLPHIN_N_TRAIN=4 DOLPHIN_N_HELD=2 OUT_DIR=results_dolphin_smoke python -u dolphin_scan.py 2>&1 | tee results_dolphin_smoke/run.log
if ! grep -q "DONE t=" results_dolphin_smoke/run.log; then
  echo "FATAL: smoke did not finish"; cp /workspace/boot.log results_dolphin/boot.log; touch DONE; sleep 25
  python - <<'PY'
import os,requests
pid=os.environ.get("RUNPOD_POD_ID");key=os.environ.get("RUNPOD_API_KEY")
if pid and key: requests.post("https://api.runpod.io/graphql",headers={"Authorization":f"Bearer {key}","Content-Type":"application/json"},json={"query":'mutation{podTerminate(input:{podId:"%s"})}'%pid},timeout=30)
PY
  exit 1
fi
echo "SMOKE_OK $(date -u)"

echo "FULL_START $(date -u)"
DOLPHIN_N_TRAIN=${N_TRAIN:-300} DOLPHIN_N_HELD=${N_HELD:-100} \
  COT_MIN=${COT_MIN:-1000} COT_MAX=${COT_MAX:-1600} POS_FRAC=${POS_FRAC:-0.08} \
  OUT_DIR=results_dolphin python -u dolphin_scan.py 2>&1 | tee results_dolphin/run.log
echo "FULL_DONE rc=$? $(date -u)"
cp /workspace/boot.log results_dolphin/boot.log; touch DONE; sleep 25
python - <<'PY'
import os,requests
pid=os.environ.get("RUNPOD_POD_ID");key=os.environ.get("RUNPOD_API_KEY")
if pid and key: requests.post("https://api.runpod.io/graphql",headers={"Authorization":f"Bearer {key}","Content-Type":"application/json"},json={"query":'mutation{podTerminate(input:{podId:"%s"})}'%pid},timeout=30); print("TERM")
PY
echo "BOOT_END $(date -u)"
