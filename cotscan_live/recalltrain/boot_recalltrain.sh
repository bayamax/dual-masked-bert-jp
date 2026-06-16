#!/bin/bash
# Self-contained recall-aware SFT pod. Designed to need NO external monitor:
#  - watchdog: the pod DELETEs ITSELF at a hard wall-clock cap (can never strand),
#  - HF heartbeat: uploads recalltrain_status.txt to HF the moment the container starts
#    (observable without the :8000 proxy; survives image-pull lag / external restarts),
#  - on completion uploads the adapter + eval json + logs to HF, then self-terminates.
# Requires env: HF_TOKEN, RUNPOD_KEY (passed via the launch env, never committed), N, EP, MAXSEC.
exec > /workspace/boot.log 2>&1
set -x
export HF_HUB_ENABLE_HF_TRANSFER=0 PYTHONUNBUFFERED=1
cd /workspace
PID="${RUNPOD_POD_ID:-unknown}"; MAXSEC="${MAXSEC:-4800}"
selfkill(){ curl -s -X DELETE -H "Authorization: Bearer $RUNPOD_KEY" "https://rest.runpod.io/v1/pods/$PID"; }
# --- watchdog: hard cap self-destruct, independent of any external monitor ---
( sleep "$MAXSEC"; echo "WATCHDOG fired after ${MAXSEC}s"; selfkill ) &

# fast install just for the heartbeat, so the marker fires within ~1min of container start
pip install -q "huggingface_hub>=0.25" 2>&1 | tail -1
cat > /workspace/mark.py <<'PY'
import sys,os,io
from huggingface_hub import HfApi
HfApi(token=os.environ.get("HF_TOKEN")).upload_file(
    path_or_fileobj=io.BytesIO(sys.argv[1].encode()),
    path_in_repo="recalltrain_status.txt", repo_id="baya1116/hypernet-sp-distill")
print("marked:",sys.argv[1])
PY
mark(){ python /workspace/mark.py "$1 @ $(date -u +%H:%M:%S)" 2>&1 | tail -1; }
mark "BOOT_STARTED N=${N:-400} EP=${EP:-1}"

pip install -q "transformers>=4.46,<4.54" "peft>=0.11" "datasets>=2.18" 2>&1 | tail -1
python - <<'PY'
from huggingface_hub import hf_hub_download
import shutil, os
R="baya1116/hypernet-sp-distill"; os.makedirs("fft_out",exist_ok=True)
for p in ["fft_out/student.pt","fft_out/pooler.pt"]: shutil.copy(hf_hub_download(R,p), p)
for p in ["build_fft_hf.py","hypernet_sp/block_recall.py","hypernet_sp/attn_export3_torch.py","hypernet_sp/attn_scenarios.py"]:
    shutil.copy(hf_hub_download(R,p), os.path.basename(p))
print("ok", os.listdir("fft_out"))
PY
python build_fft_hf.py 2>&1 | tail -2
ls -d fft_hf >/dev/null 2>&1 && mark "FFT_HF_OK" || mark "FFT_HF_MISSING"
B="https://raw.githubusercontent.com/bayamax/dual-masked-bert-jp/claude/hypernet-sp-distill-d3pyik/cotscan_live/recalltrain"
curl -fsSL "$B/recall_sft.py" -o recall_sft.py
curl -fsSL "$B/recall_eval.py" -o recall_eval.py
mark "TRAIN_START"
N=${N:-400} EP=${EP:-1} python recall_sft.py 2>&1 | tee train.log
mark "TRAIN_DONE $(grep -o 'loss=[0-9.]*' train.log | tail -1)"
python recall_eval.py 2>&1 | tee eval.log
mark "EVAL_DONE $(grep 'operand_recovered' eval.log | tail -1 | tr -d '\n' | cut -c1-120)"
# --- upload results to HF (durable, container-independent) then self-terminate ---
python - <<'PY'
import os,glob
from huggingface_hub import HfApi
api=HfApi(token=os.environ.get("HF_TOKEN")); R="baya1116/hypernet-sp-distill"
def up(l,r):
    if os.path.exists(l): api.upload_file(path_or_fileobj=l,path_in_repo=r,repo_id=R); print("up",r)
for f in glob.glob("/workspace/recall_lora/*"): up(f,"recall_lora_v2/"+os.path.basename(f))
up("/workspace/recall_eval.json","recall_eval_v2.json")
up("/workspace/recall_eval.log","recall_eval_v2.log")
up("/workspace/train.log","recall_train_v2.log")
print("HF_UPLOAD_DONE")
PY
mark "UPLOADED self-terminate"
selfkill
