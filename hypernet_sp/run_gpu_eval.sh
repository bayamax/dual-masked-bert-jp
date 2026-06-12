#!/bin/bash
# GPU evaluation job (GPUEVAL_V1) — executed INSIDE a RunPod pytorch pod.
# Pulled from the HF repo and piped to bash by the pod's docker start command.
# Env expected: HF_TOKEN, RUNPOD_API_KEY, RUNPOD_POD_ID (auto-set by RunPod).
#
# Evaluates option A (BGE-scored verbatim block recall) on GPU:
#   1) needle bench, 4 conditions (sp / qk / bge / full ceiling), 4213-tok history
#   2) composite battery v1 with recall ON   (regression: expect 12/12)
#   3) composite battery v2 with recall ON   (regression: expect 17/17)
#   4) composite battery v1 with recall OFF  (GPU-port control: expect 12/12)
# Results are committed back to the HF repo under hypernet_sp/gpu_eval/ after each
# stage, then the pod deletes itself.
set -x
cd /workspace || cd /
apt-get update -qq 2>/dev/null; apt-get install -y -qq git-lfs curl 2>/dev/null || true
git lfs install --skip-repo
git clone "https://baya1116:${HF_TOKEN}@huggingface.co/baya1116/hypernet-sp-distill" repo
cd repo || exit 1
git config user.email "gpu-eval@local"; git config user.name "gpu-eval"
pip install -q -U torch 2>&1 | tail -1
pip install -q "transformers==5.10.2" joblib scikit-learn numpy 2>&1 | tail -1
mkdir -p hypernet_sp/gpu_eval
{ echo "GPUEVAL_V1 start $(date -u)"; nvidia-smi --query-gpu=name,memory.total --format=csv; \
  python3 -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"; } \
  > hypernet_sp/gpu_eval/STATUS.txt 2>&1

push_stage () {
  cp -f needle_results.json hypernet_sp/gpu_eval/ 2>/dev/null
  echo "$1 $(date -u)" >> hypernet_sp/gpu_eval/STATUS.txt
  git add hypernet_sp/gpu_eval && git commit -qm "gpu_eval: $1" || true
  git push -q origin main || { git pull --rebase -q origin main; git push -q origin main; } || true
}
push_stage "env ready"

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
