#!/bin/bash
# Phase B job — PHASEB_POD_V1. Teacher-label generation + indexer training on a pod.
# Staged: env (V6-proven recipe) -> label smoke n=3 -> scale n=300 -> train -> upload.
# No self-delete (RUNPOD_POD_ID not injected); ends with sleep to prevent restart loops;
# the session deletes the pod on ALL_DONE/FATAL.
set -x
cd /workspace 2>/dev/null || cd /
REPO=baya1116/hypernet-sp-distill

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
snapshot_download("baya1116/hypernet-sp-distill", token=os.environ["HF_TOKEN"],
                  local_dir="/workspace/repo",
                  allow_patterns=["hypernet_sp/*.py", "runtime/*", "evals/*",
                                  "fft_out/student.pt", "build_fft_hf.py"])
PY
cd /workspace/repo || exit 1
mkdir -p hypernet_sp/phaseb
python3 - > hypernet_sp/phaseb/canary.log 2>&1 <<'PY'
import importlib, traceback, torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
try:
    importlib.import_module("transformers.models.qwen2.modeling_qwen2")
    print("CANARY_OK")
except Exception:
    traceback.print_exc()
PY
python3 build_fft_hf.py > hypernet_sp/phaseb/build_fft.log 2>&1

{ echo "PHASEB_POD_V1 start $(date -u)"; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; \
  tail -1 hypernet_sp/phaseb/canary.log; tail -1 hypernet_sp/phaseb/build_fft.log; } \
  > hypernet_sp/phaseb/STATUS.txt 2>&1

push_stage () {
  cp -f phaseb_labels_*.npz phaseb_indexer.npz hypernet_sp/phaseb/ 2>/dev/null
  echo "$1 $(date -u)" >> hypernet_sp/phaseb/STATUS.txt
  python3 -c "
from huggingface_hub import upload_folder
import os
upload_folder(repo_id='$REPO', folder_path='hypernet_sp/phaseb',
              path_in_repo='hypernet_sp/phaseb', token=os.environ['HF_TOKEN'],
              commit_message='phaseb: $1')" || true
}
push_stage "env ready (PHASEB_POD_V1)"

if ! grep -q CANARY_OK hypernet_sp/phaseb/canary.log || [ ! -d fft_hf ]; then
  push_stage "FATAL: env"; sleep infinity
fi

timeout 900 python3 hypernet_sp/phaseb_labels.py --n 3 --shard 999 \
  > hypernet_sp/phaseb/labels_smoke.log 2>&1
if ! grep -q PHASEB_LABELS_DONE hypernet_sp/phaseb/labels_smoke.log; then
  push_stage "FATAL: label smoke failed"; sleep infinity
fi
push_stage "smoke ok"

timeout 7200 python3 hypernet_sp/phaseb_labels.py --n 300 --shard 0 \
  > hypernet_sp/phaseb/labels_full.log 2>&1
push_stage "labels done (exit $?)"

timeout 3600 python3 hypernet_sp/phaseb_train.py --data 'phaseb_labels_0.npz' \
  > hypernet_sp/phaseb/train.log 2>&1
push_stage "train done (exit $?); ALL_DONE"
sleep infinity
