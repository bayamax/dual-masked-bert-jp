#!/bin/bash
# CoT-recall job — COTSCAN_POD_V1. Env recipe = PHASEB_POD_V2. Natural model-generated
# CoT (full-KV) + production-context query harvest + indexer comparison. Artifacts over
# HTTP port 8000; no HF token. Session deletes the pod on ALL_DONE/FATAL.
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
OUT=hypernet_sp/cotscan
mkdir -p $OUT
( cd $OUT && python3 -m http.server 8000 >/dev/null 2>&1 & )
status () { cp -f cot_recall_labels_*.npz $OUT/ 2>/dev/null; echo "$1 $(date -u)" >> $OUT/STATUS.txt; }

for f in cot_recall_labels.py cot_attn_scan_labels.py cot_recall_eval.py recall_query_labels.py cot_questions.json; do
  curl -sfL "$RAW/$f" -o hypernet_sp/$f
  [ -s hypernet_sp/$f ] || { status "FATAL: fetch $f"; sleep infinity; }
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
{ echo "COTSCAN_POD_V1 start $(date -u)"; nvidia-smi --query-gpu=name --format=csv,noheader; \
  tail -1 $OUT/canary.log; tail -1 $OUT/build_fft.log; } >> $OUT/STATUS.txt 2>&1
if ! grep -q CANARY_OK $OUT/canary.log || [ ! -d fft_hf ] || [ ! -f fft_out/pooler.pt ]; then
  status "FATAL: env"; sleep infinity
fi
status "env ready"

timeout 1800 python3 hypernet_sp/cot_attn_scan_labels.py --n 6 --shard 999 \
  > $OUT/labels_smoke.log 2>&1
if ! grep -q COTSCAN_LABELS_DONE $OUT/labels_smoke.log; then status "FATAL: smoke"; sleep infinity; fi
# the smoke must actually yield natural-CoT samples (model emits the value), else the
# whole battery is empty — gate on a positive sample count.
SM=$(grep -o "samples=[0-9]*" $OUT/labels_smoke.log | tail -1 | cut -d= -f2)
if [ "${SM:-0}" -lt 1 ]; then status "FATAL: smoke produced 0 natural samples"; sleep infinity; fi
status "smoke ok ($SM samples)"

timeout 12000 python3 hypernet_sp/cot_attn_scan_labels.py --n 240 --shard 0 \
  > $OUT/labels_full.log 2>&1
status "labels done (exit $?)"

timeout 4000 python3 hypernet_sp/cot_attn_scan_labels.py --n 80 --shard 1 --held \
  > $OUT/labels_heldout.log 2>&1
status "heldout labels done (exit $?)"

timeout 2400 python3 hypernet_sp/cot_recall_eval.py --data cot_recall_labels_0.npz \
  --evaldata cot_recall_labels_1.npz > $OUT/eval.log 2>&1
status "eval done (exit $?); ALL_DONE"
sleep infinity
