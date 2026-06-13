#!/bin/bash
# Gated-recall acceptance job — GATED_POD_V1. Env recipe = PHASEB_POD_V2.
# Pulls the patched app_session_torch.py + trigger_gate.py + gated_needle_test.py
# from this branch over the snapshot repo's own files, runs always-vs-gated.
# Artifacts over HTTP port 8000; no HF token. Session deletes the pod on ALL_DONE/FATAL.
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
                                  "evals/*", "build_fft_hf.py"])
PY
cd /workspace/repo || exit 1
OUT=hypernet_sp/gated
mkdir -p $OUT
( cd $OUT && python3 -m http.server 8000 >/dev/null 2>&1 & )
status () { cp -f gated_needle_results.json $OUT/ 2>/dev/null; echo "$1 $(date -u)" >> $OUT/STATUS.txt; }

# patched runtime + gate + test + head, fetched fresh from this branch
curl -sfL "$RAW/app_session_torch.py" -o hypernet_sp/app_session_torch.py
curl -sfL "$RAW/trigger_gate.py" -o hypernet_sp/trigger_gate.py
curl -sfL "$RAW/gated_needle_test.py" -o hypernet_sp/gated_needle_test.py
curl -sfL "$RAW/results_v3/trigger_head_v3_gate.npz" -o trigger_head_v3_gate.npz
for f in hypernet_sp/app_session_torch.py hypernet_sp/trigger_gate.py \
         hypernet_sp/gated_needle_test.py trigger_head_v3_gate.npz; do
  [ -s "$f" ] || { status "FATAL: fetch $f"; sleep infinity; }
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

{ echo "GATED_POD_V1 start $(date -u)"; nvidia-smi --query-gpu=name --format=csv,noheader; \
  tail -1 $OUT/canary.log; tail -1 $OUT/build_fft.log; } >> $OUT/STATUS.txt 2>&1
if ! grep -q CANARY_OK $OUT/canary.log || [ ! -d fft_hf ] || [ ! -f fft_out/pooler.pt ]; then
  status "FATAL: env"; sleep infinity
fi
status "env ready"

export SPCHAT_DEVICE=cuda
timeout 1800 python3 hypernet_sp/gated_needle_test.py --smoke > $OUT/smoke.log 2>&1
if ! grep -q NEEDLE_GATED_DONE $OUT/smoke.log; then status "FATAL: smoke"; sleep infinity; fi
status "smoke ok"

timeout 7200 python3 hypernet_sp/gated_needle_test.py > $OUT/full.log 2>&1
status "full done (exit $?); ALL_DONE"
sleep infinity
