#!/bin/bash
# GPU evaluation job — GPUEVAL_V4. History: V1 wrong-wheel/no-fft_hf; V2 torch 2.11
# (cu128 index has no 2.12) broke transformers 5.10.2; V3's pip loop was pipe-masked
# (tail's exit code) so the pin never installed and the image's torch 2.4.1 lacked
# torch.float8_e8m0fnu. V4 installs torch==2.12.0 from cu126 (runs on any driver >=12.6,
# covers all community machines seen), VERIFIES the install + fp8 attr + cuda before
# proceeding, and gates stages on CANARY_OK.
# Executed INSIDE a RunPod pod. Env: HF_TOKEN, RUNPOD_API_KEY, RUNPOD_POD_ID.
set -x
export HF_HUB_ENABLE_HF_TRANSFER=0
cd /workspace 2>/dev/null || cd /
REPO=baya1116/hypernet-sp-distill

for IDX in cu126 cu130; do
  pip install -q --force-reinstall "torch==2.12.0" --index-url "https://download.pytorch.org/whl/$IDX" > /tmp/pip_torch.log 2>&1
  TCHK=$(python3 -c "import torch; print('OK' if hasattr(torch,'float8_e8m0fnu') and torch.cuda.is_available() else 'BAD', torch.__version__)" 2>/dev/null)
  echo "torch attempt $IDX -> $TCHK"
  case "$TCHK" in OK*) break;; esac
done
pip install -q "transformers==5.10.2" huggingface_hub joblib scikit-learn numpy > /tmp/pip_rest.log 2>&1

python3 - <<'PY'
from huggingface_hub import snapshot_download
import os
snapshot_download("baya1116/hypernet-sp-distill", token=os.environ["HF_TOKEN"],
                  local_dir="/workspace/repo",
                  allow_patterns=["hypernet_sp/*", "runtime/*", "evals/*",
                                  "fft_out/student.pt", "fft_out/pooler.pt",
                                  "build_fft_hf.py"])
PY
cd /workspace/repo || exit 1
mkdir -p hypernet_sp/gpu_eval
python3 - > hypernet_sp/gpu_eval/canary.log 2>&1 <<'PY'
import traceback, torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
try:
    import transformers
    print("transformers", transformers.__version__)
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa
    from transformers.models.qwen2 import modeling_qwen2          # noqa
    print("CANARY_OK")
except Exception:
    traceback.print_exc()
PY
python3 build_fft_hf.py > hypernet_sp/gpu_eval/build_fft.log 2>&1

{ echo "GPUEVAL_V4 start $(date -u)"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv; \
  python3 -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"; \
  tail -1 hypernet_sp/gpu_eval/canary.log; tail -1 hypernet_sp/gpu_eval/build_fft.log; } > hypernet_sp/gpu_eval/STATUS.txt 2>&1

push_stage () {
  cp -f needle_results.json hypernet_sp/gpu_eval/ 2>/dev/null
  echo "$1 $(date -u)" >> hypernet_sp/gpu_eval/STATUS.txt
  python3 -c "
from huggingface_hub import upload_folder
import os
upload_folder(repo_id='$REPO', folder_path='hypernet_sp/gpu_eval',
              path_in_repo='hypernet_sp/gpu_eval', token=os.environ['HF_TOKEN'],
              commit_message='gpu_eval: $1')" || true
}
push_stage "env ready (GPUEVAL_V4)"

if ! grep -q CANARY_OK hypernet_sp/gpu_eval/canary.log || [ ! -d fft_hf ]; then
  push_stage "FATAL: canary or fft_hf build failed - see canary.log / build_fft.log; aborting"
  curl -s -X DELETE "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID}" \
    -H "Authorization: Bearer ${RUNPOD_API_KEY}"
  sleep 120; exit 1
fi

SPCHAT_DEVICE=cuda timeout 5400 python3 hypernet_sp/needle_recall_test.py \
  > hypernet_sp/gpu_eval/needle.log 2>&1
push_stage "stage1 needle done (exit $?)"

SPCHAT_DEVICE=cuda SPCHAT_BLOCK_RECALL=bge timeout 5400 python3 hypernet_sp/composite_test.py \
  > hypernet_sp/gpu_eval/comp1_bge.log 2>&1
push_stage "stage2 composite-v1 recall-ON done (exit $?)"

SPCHAT_DEVICE=cuda SPCHAT_BLOCK_RECALL=bge timeout 5400 python3 hypernet_sp/composite_test2.py \
  > hypernet_sp/gpu_eval/comp2_bge.log 2>&1
push_stage "stage3 composite-v2 recall-ON done (exit $?)"

SPCHAT_DEVICE=cuda timeout 5400 python3 hypernet_sp/composite_test.py \
  > hypernet_sp/gpu_eval/comp1_off.log 2>&1
push_stage "stage4 composite-v1 control-OFF done (exit $?); ALL_DONE"

curl -s -X DELETE "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID}" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}"
sleep 120
