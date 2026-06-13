#!/bin/bash
set +e
exec > /workspace/boot.log 2>&1
echo "BOOT_START $(date -u)"
export HF_HUB_ENABLE_HF_TRANSFER=0
RID="baya1116/hypernet-sp-distill"
GH="https://raw.githubusercontent.com/bayamax/dual-masked-bert-jp/e664ec9bb30b5a9a536439f0f87f498d887e2a31/trigger_experiment"
OUT="trigger_experiment/results_taskeval"

pip install -q -U "huggingface_hub>=0.25" "transformers>=4.46,<4.54" "scikit-learn>=1.3" \
    requests 2>&1 | tail -2
echo "PIP_DONE $(date -u)"
mkdir -p /work && cd /work

python - <<'PY'
import os, shutil
from huggingface_hub import hf_hub_download
R="baya1116/hypernet-sp-distill"; tok=os.environ["HF_TOKEN"]
for p in ["hypernet_sp/attn_export3_torch.py","hypernet_sp/attn_scenarios.py",
          "hypernet_sp/block_recall.py","hypernet_sp/phaseb_data.py",
          "build_fft_hf.py","fft_out/student.pt","fft_out/pooler.pt"]:
    f=hf_hub_download(R,p,token=tok)
    dst=p if p.startswith("fft_out/") else os.path.basename(p)
    os.makedirs(os.path.dirname(dst) or ".",exist_ok=True); shutil.copy(f,dst)
    print("got",dst,os.path.getsize(dst))
print("HF_DOWNLOAD_DONE")
PY
for f in cot_recall_labels.py cot_recall_natural.py cot_recall_eval.py recall_query_labels.py cot_questions.json; do
  curl -sfL "$GH/$f" -o "$f" && echo "got $f ($(wc -c <$f)B)" || { echo "FATAL fetch $f"; }
done
echo "$TRAIN_PY_B64" | base64 -d > train_components.py
echo "$TASK_PY_B64"  | base64 -d > cotscan_task_eval.py
wc -l train_components.py cotscan_task_eval.py
echo "BUILD_FFT $(date -u)"; python build_fft_hf.py 2>&1 | tail -2; echo "BUILD_FFT_DONE $(date -u)"

# uploader
python - <<'PY' &
import time, os
from huggingface_hub import HfApi
api=HfApi(token=os.environ["HF_TOKEN"]); R="baya1116/hypernet-sp-distill"; O="trigger_experiment/results_taskeval"
while True:
    for p,rep in [("results_taskeval/live.log",O+"/live.log"),
                  ("results_taskeval/results.json",O+"/results.json"),
                  ("components_summary.json",O+"/components_summary.json"),
                  ("eval_natural.log",O+"/eval_natural.log")]:
        if os.path.exists(p):
            try: api.upload_file(path_or_fileobj=p,path_in_repo=rep,repo_id=R)
            except Exception as e: print("up_err",e,flush=True)
    if os.path.exists("DONE"): break
    time.sleep(25)
PY

mkdir -p results_taskeval
echo "LABELS_START $(date -u)"
python cot_recall_natural.py --n ${LBL_N0:-80} --shard 0 --maxgen ${MAXGEN:-2500} > results_taskeval/labels0.log 2>&1
echo "labels0 rc=$? $(grep -o 'samples=[0-9]*' results_taskeval/labels0.log | tail -1)"
python cot_recall_natural.py --n ${LBL_N1:-30} --shard 1 --maxgen ${MAXGEN:-2500} --held > results_taskeval/labels1.log 2>&1
echo "labels1 rc=$? $(grep -o 'samples=[0-9]*' results_taskeval/labels1.log | tail -1)"

echo "TRAIN_COMPONENTS $(date -u)"
python train_components.py > results_taskeval/train_components.log 2>&1
echo "train rc=$?"; tail -5 results_taskeval/train_components.log

# reference: also run the authors' eval to reproduce AUC/top-2 on the same labels
python cot_recall_eval.py --data cot_recall_natural_0.npz --evaldata cot_recall_natural_1.npz \
  > eval_natural.log 2>&1; echo "ref_eval rc=$?"

echo "TASK_EVAL_START $(date -u)"
TASK_SEEDS=${TASK_SEEDS:-3} OUT_DIR=results_taskeval python cotscan_task_eval.py
echo "TASK_EVAL_DONE rc=$? $(date -u)"
touch DONE

python - <<'PY'
import os
from huggingface_hub import HfApi
api=HfApi(token=os.environ["HF_TOKEN"]); R="baya1116/hypernet-sp-distill"; O="trigger_experiment/results_taskeval"
for p,name in [("results_taskeval/results.json","results.json"),("results_taskeval/live.log","live.log"),
               ("components_summary.json","components_summary.json"),("eval_natural.log","eval_natural.log"),
               ("results_taskeval/train_components.log","train_components.log"),
               ("results_taskeval/labels0.log","labels0.log"),("results_taskeval/labels1.log","labels1.log"),
               ("/workspace/boot.log","boot.log")]:
    if os.path.exists(p):
        try: api.upload_file(path_or_fileobj=p,path_in_repo=f"{O}/{name}",repo_id=R); print("flushed",name)
        except Exception as e: print("flush_err",name,e)
PY
echo "FLUSH_DONE $(date -u)"
python - <<'PY'
import os, requests
pid=os.environ.get("RUNPOD_POD_ID"); key=os.environ.get("RUNPOD_API_KEY")
if pid and key:
    requests.post("https://api.runpod.io/graphql",headers={"Authorization":f"Bearer {key}","Content-Type":"application/json"},
        json={"query":'mutation{podTerminate(input:{podId:"%s"})}'%pid},timeout=30); print("TERMINATE",pid)
PY
echo "BOOT_END $(date -u)"
