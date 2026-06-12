#!/bin/bash
# GPU evaluation job — GPUEVAL_V2 (V1 failed: image fell back to torch cu130 wheel vs
# 12.8 driver -> cuda False; and the HF repo has no fft_hf/, it must be rebuilt from
# fft_out/student.pt via build_fft_hf.py; git-lfs clone was silently skipping LFS).
# Executed INSIDE a RunPod pod. Env: HF_TOKEN, RUNPOD_API_KEY, RUNPOD_POD_ID.
set -x
export HF_HUB_ENABLE_HF_TRANSFER=0
cd /workspace 2>/dev/null || cd /
REPO=baya1116/hypernet-sp-distill

pip install -q --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -1
pip install -q "transformers==5.10.2" huggingface_hub joblib scikit-learn numpy 2>&1 | tail -1

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
python3 build_fft_hf.py > hypernet_sp/gpu_eval/build_fft.log 2>&1

{ echo "GPUEVAL_V2 start $(date -u)"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv; \
  python3 -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"; \
  tail -1 hypernet_sp/gpu_eval/build_fft.log; } > hypernet_sp/gpu_eval/STATUS.txt 2>&1

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
push_stage "env ready (GPUEVAL_V2)"

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
