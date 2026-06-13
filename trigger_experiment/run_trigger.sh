#!/bin/bash
# Trigger-head experiment job — TRIGGER_POD_V1. Env recipe = PHASEB_POD_V2 (proven).
# Differences: scripts fetched from GitHub raw (this branch), artifacts served over
# HTTP on port 8000 (RunPod proxy) instead of HF upload — no HF token required.
# No self-delete (RUNPOD_POD_ID not injected); ends with sleep to prevent restart
# loops; the session deletes the pod on ALL_DONE/FATAL.
set -x
cd /workspace 2>/dev/null || cd /
RAW=https://raw.githubusercontent.com/bayamax/dual-masked-bert-jp/claude/huggingface-model-status-ypclnu/trigger_experiment

for IDX in cu126 cu130; do
  pip install -q --force-reinstall "torch==2.12.0" --index-url "https://download.pytorch.org/whl/$IDX" > /tmp/pip_torch.log 2>&1
  TCHK=$(python3 -c "import torch; print('OK' if hasattr(torch,'float8_e8m0fnu') and torch.cuda.is_available() else 'BAD')" 2>/dev/null)
  case "$TCHK" in OK*) break;; esac
done
pip uninstall -q -y flash-attn flash_attn apex xformers torchvision torchaudio 2>/dev/null
pip install -q "transformers==5.10.2" huggingface_hub joblib scikit-learn numpy > /tmp/pip_rest.log 2>&1

python3 - <<'PY'
from huggingface_hub import snapshot_download
import os
snapshot_download("baya1116/hypernet-sp-distill", token=os.environ.get("HF_TOKEN"),
                  local_dir="/workspace/repo",
                  allow_patterns=["hypernet_sp/*.py", "runtime/*",
                                  "fft_out/student.pt", "build_fft_hf.py"])
PY
cd /workspace/repo || exit 1
OUT=hypernet_sp/trigger
mkdir -p $OUT
( cd $OUT && python3 -m http.server 8000 >/dev/null 2>&1 & )

status () {
  cp -f trigger_labels_*.npz trigger_head.npz $OUT/ 2>/dev/null
  echo "$1 $(date -u)" >> $OUT/STATUS.txt
}

curl -sfL "$RAW/trigger_labels.py" -o hypernet_sp/trigger_labels.py
curl -sfL "$RAW/trigger_train.py" -o hypernet_sp/trigger_train.py
if [ ! -s hypernet_sp/trigger_labels.py ] || [ ! -s hypernet_sp/trigger_train.py ]; then
  status "FATAL: script fetch"; sleep infinity
fi

python3 - > $OUT/canary.log 2>&1 <<'PY'
import importlib, traceback, torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
try:
    importlib.import_module("transformers.models.qwen2.modeling_qwen2")
    print("CANARY_OK")
except Exception:
    traceback.print_exc()
PY
python3 build_fft_hf.py > $OUT/build_fft.log 2>&1

{ echo "TRIGGER_POD_V1 start $(date -u)"; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; \
  tail -1 $OUT/canary.log; tail -1 $OUT/build_fft.log; } >> $OUT/STATUS.txt 2>&1

if ! grep -q CANARY_OK $OUT/canary.log || [ ! -d fft_hf ]; then
  status "FATAL: env"; sleep infinity
fi
status "env ready (TRIGGER_POD_V1)"

timeout 1200 python3 hypernet_sp/trigger_labels.py --n 2 --shard 999 \
  > $OUT/labels_smoke.log 2>&1
if ! grep -q TRIGGER_LABELS_DONE $OUT/labels_smoke.log; then
  status "FATAL: label smoke failed"; sleep infinity
fi
status "smoke ok"

timeout 9000 python3 hypernet_sp/trigger_labels.py --n 200 --shard 0 \
  > $OUT/labels_full.log 2>&1
status "labels done (exit $?)"

timeout 3600 python3 hypernet_sp/trigger_labels.py --n 60 --shard 1 --heldout \
  > $OUT/labels_heldout.log 2>&1
status "heldout labels done (exit $?)"

timeout 1800 python3 hypernet_sp/trigger_train.py --data trigger_labels_0.npz \
  --evaldata trigger_labels_1.npz --out trigger_head.npz \
  > $OUT/train.log 2>&1
status "train done (exit $?); ALL_DONE"
sleep infinity
