#!/bin/bash
# RunPod bootstrap: pull code+weights from HF, build fft_hf, run cotscan_live, stream
# results to HF, then self-terminate. No SSH needed; progress is visible via HF uploads.
set +e
exec > /workspace/boot.log 2>&1
echo "BOOT_START $(date -u)"
export HF_HUB_ENABLE_HF_TRANSFER=0
RID="baya1116/hypernet-sp-distill"
OUTREPO="trigger_experiment/results_cotscan_live"

pip install -q -U "huggingface_hub>=0.25" "transformers>=4.46,<4.54" requests 2>&1 | tail -3
echo "PIP_DONE $(date -u)"

mkdir -p /work/hypernet_sp && cd /work/hypernet_sp

python - <<'PY'
import os, shutil
from huggingface_hub import hf_hub_download
R = "baya1116/hypernet-sp-distill"; tok = os.environ["HF_TOKEN"]
files = ["hypernet_sp/block_recall.py","hypernet_sp/attn_export3_torch.py",
         "hypernet_sp/attn_scenarios.py","build_fft_hf.py",
         "fft_out/student.pt","fft_out/pooler.pt"]
for p in files:
    f = hf_hub_download(R, p, token=tok)
    dst = p if p.startswith("fft_out/") else os.path.basename(p)
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    shutil.copy(f, dst)
    print("got", dst, os.path.getsize(dst))
print("DOWNLOAD_DONE")
PY
echo "DL_DONE $(date -u)"

echo "$COTSCAN_PY_B64" | base64 -d > cotscan_live.py
wc -l cotscan_live.py

echo "BUILD_FFT_START $(date -u)"
python build_fft_hf.py
echo "BUILD_FFT_DONE $(date -u)"

# background uploader: stream live.log + results.json to HF every 20s
python - <<'PY' &
import time, os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"]); R="baya1116/hypernet-sp-distill"
OUT="trigger_experiment/results_cotscan_live"
while True:
    for fn in ["live.log","results.json"]:
        p = f"results_cotscan_live/{fn}"
        if os.path.exists(p):
            try: api.upload_file(path_or_fileobj=p, path_in_repo=f"{OUT}/{fn}", repo_id=R)
            except Exception as e: print("up_err", e, flush=True)
    if os.path.exists("results_cotscan_live/.done"): break
    time.sleep(20)
print("UPLOADER_EXIT", flush=True)
PY
UPPID=$!

echo "RUN_START $(date -u)"
OUT_DIR=results_cotscan_live python cotscan_live.py
echo "RUN_DONE $(date -u) rc=$?"
touch results_cotscan_live/.done

# final flush
python - <<'PY'
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"]); R="baya1116/hypernet-sp-distill"
OUT="trigger_experiment/results_cotscan_live"
for p,name in [("results_cotscan_live/results.json","results.json"),
               ("results_cotscan_live/live.log","live.log"),
               ("/workspace/boot.log","boot.log")]:
    if os.path.exists(p):
        try: api.upload_file(path_or_fileobj=p, path_in_repo=f"{OUT}/{name}", repo_id=R); print("flushed", name)
        except Exception as e: print("flush_err", name, e)
PY
echo "FLUSH_DONE $(date -u)"

# self-terminate the pod
python - <<'PY'
import os, requests
pid = os.environ.get("RUNPOD_POD_ID"); key = os.environ.get("RUNPOD_API_KEY")
if pid and key:
    q = 'mutation{podTerminate(input:{podId:"%s"})}' % pid
    r = requests.post("https://api.runpod.io/graphql",
                      headers={"Authorization": f"Bearer {key}", "Content-Type":"application/json"},
                      json={"query": q}, timeout=30)
    print("TERMINATE", r.status_code, r.text[:200])
PY
echo "BOOT_END $(date -u)"
