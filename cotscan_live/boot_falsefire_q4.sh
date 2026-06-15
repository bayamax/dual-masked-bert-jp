#!/bin/bash
# RunPod bootstrap: 4-bit recall FALSE-FIRE mitigation campaign.
#   Stage A: train a 4-bit contrastive gate (generation-distribution hard negatives) -> gate.npz
#   Stage B: 4-bit end-to-end threshold sweep, OLD (fp32 ondevice) gate vs NEW contrastive gate
# Streams live.log/results.json to HF; self-terminates. No secrets in this file (env only).
set +e
exec > /workspace/boot.log 2>&1
echo "BOOT_START $(date -u)"
export HF_HUB_ENABLE_HF_TRANSFER=0
R="baya1116/hypernet-sp-distill"

pip install -q -U "huggingface_hub>=0.25" "transformers>=4.46,<4.54" accelerate bitsandbytes \
    sentence-transformers scikit-learn datasets requests 2>&1 | tail -5
echo "PIP_DONE $(date -u)"

mkdir -p /work/hypernet_sp && cd /work/hypernet_sp

# early heartbeat uploader: stream boot.log throughout so failures are visible even pre-results
python - <<'PY' &
import time, os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"]); R = "baya1116/hypernet-sp-distill"
while not os.path.exists("ALL_DONE"):
    if os.path.exists("/workspace/boot.log"):
        try: api.upload_file(path_or_fileobj="/workspace/boot.log",
                             path_in_repo="trigger_experiment/results_gate_v3_q4/boot.log", repo_id=R)
        except Exception as e: print("boot_up_err", e, flush=True)
    time.sleep(20)
PY

python - <<'PY'
import os, shutil
from huggingface_hub import hf_hub_download
R = "baya1116/hypernet-sp-distill"; tok = os.environ["HF_TOKEN"]
deps = ["hypernet_sp/block_recall.py", "hypernet_sp/attn_export3_torch.py",
        "hypernet_sp/attn_scenarios.py", "build_fft_hf.py",
        "fft_out/student.pt", "fft_out/pooler.pt",
        "trigger_experiment/ondevice_recall/bge_head.npz",
        "trigger_experiment/ondevice_recall/gate.npz"]
scripts = ["mix_scan.py", "gate_contrastive.py", "gate_v3.py", "retrain_q4.py",
           "gate_v3_q4.py", "recall_gen.py", "cot_recall_eval.py"]
for p in deps:
    f = hf_hub_download(R, p, token=tok)
    if p.startswith("fft_out/"):           dst = p
    elif p.endswith("bge_head.npz"):        dst = "bge_head.npz"
    elif p.endswith("ondevice_recall/gate.npz"): dst = "gate_ondevice.npz"
    else:                                   dst = os.path.basename(p)
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True); shutil.copy(f, dst); print("got", dst)
for p in scripts:
    f = hf_hub_download(R, "trigger_experiment/_run_ff_q4/" + p, token=tok)
    shutil.copy(f, p); print("got", p)
print("DOWNLOAD_DONE")
PY
echo "DL_DONE $(date -u)"

echo "BUILD_FFT_START $(date -u)"
python build_fft_hf.py
echo "BUILD_FFT_DONE $(date -u)"

# background uploader: stream live.log + results.json for all stages to HF
python - <<'PY' &
import time, os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"]); R = "baya1116/hypernet-sp-distill"
dirs = {"results_gate_v3_q4": "trigger_experiment/results_gate_v3_q4",
        "results_recall_q4_oldgate": "trigger_experiment/results_recall_q4_oldgate",
        "results_recall_q4_newgate": "trigger_experiment/results_recall_q4_newgate"}
while True:
    for ld, rd in dirs.items():
        for fn in ["live.log", "results.json"]:
            p = f"{ld}/{fn}"
            if os.path.exists(p):
                try: api.upload_file(path_or_fileobj=p, path_in_repo=f"{rd}/{fn}", repo_id=R)
                except Exception as e: print("up_err", e, flush=True)
    if os.path.exists("ALL_DONE"): break
    time.sleep(25)
print("UPLOADER_EXIT", flush=True)
PY

# ---------------- STAGE A: 4-bit contrastive gate ----------------
echo "STAGE_A_START $(date -u)"
OUT_DIR=results_gate_v3_q4 ART_DIR=artifacts_v3_q4 python gate_v3_q4.py
echo "STAGE_A_DONE rc=$? $(date -u)"
python - <<'PY'
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"]); R = "baya1116/hypernet-sp-distill"
p = "artifacts_v3_q4/gate.npz"
if os.path.exists(p):
    api.upload_file(path_or_fileobj=p, path_in_repo="trigger_experiment/results_gate_v3_q4/gate.npz", repo_id=R)
    print("GATE_UPLOADED")
PY
cp artifacts_v3_q4/gate.npz gate_v3q4.npz 2>/dev/null

# ---------------- STAGE B: 4-bit threshold sweep ----------------
echo "STAGE_B_START $(date -u)"
# OLD fp32 ondevice gate at the known 4-bit operating band
OUT_DIR=results_recall_q4_oldgate GATE_NPZ=gate_ondevice.npz Q4=1 GEN_N=4 \
  ARMS="SP,RC_bge@-3.0,RC_bge@-3.5,RC_bge@-4.0" python recall_gen.py
echo "OLDGATE_DONE rc=$? $(date -u)"
# NEW contrastive gate: center the sweep on its saved 90%-recall operating point
NEWARMS=$(python - <<'PY'
import json
try:
    t = json.load(open("results_gate_v3_q4/results.json")).get("saved_thresh", 0.0)
except Exception:
    t = 0.0
arms = ["RC_bge@%.2f" % (t + d) for d in (-1.0, -0.5, 0.0, 0.5)]
print("SP," + ",".join(arms))
PY
)
echo "NEWARMS=$NEWARMS"
OUT_DIR=results_recall_q4_newgate GATE_NPZ=gate_v3q4.npz Q4=1 GEN_N=4 \
  ARMS="$NEWARMS" python recall_gen.py
echo "NEWGATE_DONE rc=$? $(date -u)"

touch ALL_DONE
sleep 30
# final flush (results + boot.log)
python - <<'PY'
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"]); R = "baya1116/hypernet-sp-distill"
for ld in ["results_gate_v3_q4", "results_recall_q4_oldgate", "results_recall_q4_newgate"]:
    for fn in ["live.log", "results.json"]:
        p = f"{ld}/{fn}"
        if os.path.exists(p):
            try: api.upload_file(path_or_fileobj=p, path_in_repo=f"trigger_experiment/{ld}/{fn}", repo_id=R)
            except Exception as e: print("flush_err", e)
if os.path.exists("/workspace/boot.log"):
    try: api.upload_file(path_or_fileobj="/workspace/boot.log", path_in_repo="trigger_experiment/results_gate_v3_q4/boot.log", repo_id=R)
    except Exception as e: print("boot_flush_err", e)
print("FLUSH_DONE")
PY

# self-terminate
python - <<'PY'
import os, requests
pid = os.environ.get("RUNPOD_POD_ID"); key = os.environ.get("RUNPOD_API_KEY")
if pid and key:
    q = 'mutation{podTerminate(input:{podId:"%s"})}' % pid
    r = requests.post("https://api.runpod.io/graphql",
                      headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                      json={"query": q}, timeout=30)
    print("TERMINATE", r.status_code, r.text[:200])
PY
echo "BOOT_END $(date -u)"
