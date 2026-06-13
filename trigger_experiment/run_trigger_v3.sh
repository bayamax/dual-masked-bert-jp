#!/bin/bash
# Trigger-head V2 job — TRIGGER_V3_POD. Production-context (SP+window) feature
# harvesting + head retraining + V1-head transfer check. Env recipe = PHASEB_POD_V2.
# Artifacts over HTTP port 8000 (RunPod proxy); no HF token. No self-delete; the
# session deletes the pod on ALL_DONE/FATAL.
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
                                  "fft_out/student.pt", "fft_out/pooler.pt",
                                  "build_fft_hf.py"])
PY
cd /workspace/repo || exit 1
OUT=hypernet_sp/trigger_v3
mkdir -p $OUT
( cd $OUT && python3 -m http.server 8000 >/dev/null 2>&1 & )

status () {
  cp -f trigger_labels_v3_*.npz trigger_head_v3.npz $OUT/ 2>/dev/null
  echo "$1 $(date -u)" >> $OUT/STATUS.txt
}

curl -sfL "$RAW/trigger_labels.py" -o hypernet_sp/trigger_labels.py
curl -sfL "$RAW/trigger_labels_v3.py" -o hypernet_sp/trigger_labels_v3.py
curl -sfL "$RAW/battery_ext.json" -o hypernet_sp/battery_ext.json
curl -sfL "$RAW/trigger_train.py" -o hypernet_sp/trigger_train.py
curl -sfL "$RAW/results_sp/trigger_head_sp.npz" -o trigger_head_v2.npz
for f in hypernet_sp/trigger_labels.py hypernet_sp/trigger_labels_v3.py \
         hypernet_sp/trigger_train.py trigger_head_v2.npz hypernet_sp/battery_ext.json; do
  [ -s "$f" ] || { status "FATAL: script fetch $f"; sleep infinity; }
done

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

{ echo "TRIGGER_V3_POD start $(date -u)"; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; \
  tail -1 $OUT/canary.log; tail -1 $OUT/build_fft.log; } >> $OUT/STATUS.txt 2>&1

if ! grep -q CANARY_OK $OUT/canary.log || [ ! -d fft_hf ] || [ ! -f fft_out/pooler.pt ]; then
  status "FATAL: env"; sleep infinity
fi
status "env ready (TRIGGER_V3_POD)"

timeout 1200 python3 hypernet_sp/trigger_labels_v3.py --n 2 --shard 999 \
  > $OUT/labels_smoke.log 2>&1
if ! grep -q TRIGGER_LABELS_DONE $OUT/labels_smoke.log; then
  status "FATAL: label smoke failed"; sleep infinity
fi
status "smoke ok"

timeout 9000 python3 hypernet_sp/trigger_labels_v3.py --n 200 --shard 0 \
  > $OUT/labels_full.log 2>&1
status "labels done (exit $?)"

timeout 3600 python3 hypernet_sp/trigger_labels_v3.py --n 60 --shard 1 --heldout \
  > $OUT/labels_heldout.log 2>&1
status "heldout labels done (exit $?)"

timeout 1800 python3 hypernet_sp/trigger_train.py --data trigger_labels_v3_0.npz \
  --evaldata trigger_labels_v3_1.npz --prevhead trigger_head_v2.npz \
  --out trigger_head_v3.npz > $OUT/train.log 2>&1
status "train done (exit $?); ALL_DONE"
sleep infinity
