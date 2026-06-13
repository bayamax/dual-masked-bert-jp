#!/bin/bash
set +e
exec > /workspace/boot.log 2>&1
echo "BOOT_START $(date -u)"
export HF_HUB_ENABLE_HF_TRANSFER=0
O="trigger_experiment/results_gsm_gate"
pip install -q -U "huggingface_hub>=0.25" "transformers>=4.46,<4.54" "datasets>=2.18" requests 2>&1 | tail -2
echo "PIP_DONE $(date -u)"
mkdir -p /work && cd /work && mkdir -p results_gsm_gate
python - <<'PY'
import os, shutil
from huggingface_hub import hf_hub_download
R="baya1116/hypernet-sp-distill"; tok=os.environ["HF_TOKEN"]
for p in ["hypernet_sp/attn_export3_torch.py","hypernet_sp/attn_scenarios.py",
          "hypernet_sp/block_recall.py","build_fft_hf.py","fft_out/student.pt","fft_out/pooler.pt"]:
    f=hf_hub_download(R,p,token=tok); dst=p if p.startswith("fft_out/") else os.path.basename(p)
    os.makedirs(os.path.dirname(dst) or ".",exist_ok=True); shutil.copy(f,dst)
f=hf_hub_download(R,"trigger_experiment/recall_kit_v4/artifacts/gate.npz",token=tok)
shutil.copy(f,"dolphin_gate.npz"); print("got dolphin_gate.npz")
print("HF_DONE")
PY
echo "$GSM_PY_B64" | base64 -d > cotscan_gsm8k.py
echo "BUILD_FFT $(date -u)"; python build_fft_hf.py 2>&1 | tail -1; echo "BUILD_FFT_DONE $(date -u)"
python - <<'PY' &
import time, os
from huggingface_hub import HfApi
api=HfApi(token=os.environ["HF_TOKEN"]); R="baya1116/hypernet-sp-distill"; O="trigger_experiment/results_gsm_gate"
while True:
    for fn in ["live.log","results.json"]:
        p=f"results_gsm_gate/{fn}"
        if os.path.exists(p):
            try: api.upload_file(path_or_fileobj=p,path_in_repo=f"{O}/{fn}",repo_id=R)
            except Exception as e: print("up",e,flush=True)
    if os.path.exists("results_gsm_gate/.done"): break
    time.sleep(20)
PY
echo "RUN $(date -u)"
GSM8K_N=30 ARMS=OFF,TURN,GATE GATE_HEAD=dolphin_gate.npz GATE_THRESH=${GATE_THRESH:-0} \
  OUT_DIR=results_gsm_gate python -u cotscan_gsm8k.py
echo "RUN_DONE rc=$? $(date -u)"; touch results_gsm_gate/.done
cp /workspace/boot.log results_gsm_gate/boot.log
python - <<'PY'
import os
from huggingface_hub import HfApi
api=HfApi(token=os.environ["HF_TOKEN"]); R="baya1116/hypernet-sp-distill"; O="trigger_experiment/results_gsm_gate"
for fn in ["results.json","live.log","boot.log"]:
    p=f"results_gsm_gate/{fn}"
    if os.path.exists(p):
        try: api.upload_file(path_or_fileobj=p,path_in_repo=f"{O}/{fn}",repo_id=R); print("flush",fn)
        except Exception as e: print("ferr",fn,e)
PY
python - <<'PY'
import os, requests
pid=os.environ.get("RUNPOD_POD_ID");key=os.environ.get("RUNPOD_API_KEY")
if pid and key: requests.post("https://api.runpod.io/graphql",headers={"Authorization":f"Bearer {key}","Content-Type":"application/json"},json={"query":'mutation{podTerminate(input:{podId:"%s"})}'%pid},timeout=30); print("TERM")
PY
echo "BOOT_END $(date -u)"
