#!/bin/bash
# RunPod bootstrap: relevance-FLOOR confirmation for the BGE-on-demand recall path (4-bit).
# Confirmed direction: keep the proven gate + BGE retriever; suppress false-fire by only INJECTING
# when the top BGE retrieval score clears a floor (casual turns have no truly-relevant evicted
# block -> low score -> skip). Runs ONE recall_gen sweep; arms/gate/out come from env so floors
# (read off the Stage-B tscore split) are filled at launch. Streams to HF; self-terminates.
set +e
exec > /workspace/boot.log 2>&1
echo "BOOT_START $(date -u)"
export HF_HUB_ENABLE_HF_TRANSFER=0
R="baya1116/hypernet-sp-distill"
RG_OUT="${RG_OUT:-results_floor_q4}"
RG_GATE="${RG_GATE:-gate_ondevice.npz}"
RG_ARMS="${RG_ARMS:-SP,RC_bge@-3.5}"
RG_N="${RG_N:-5}"
RG_TURNCAP="${RG_TURNCAP:-90}"

pip install -q -U "huggingface_hub>=0.25" "transformers>=4.46,<4.54" accelerate bitsandbytes \
    sentence-transformers scikit-learn datasets requests 2>&1 | tail -5
echo "PIP_DONE $(date -u)"
mkdir -p /work/hypernet_sp && cd /work/hypernet_sp

# heartbeat: stream boot.log throughout
python - <<'PY' &
import time, os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"]); R = "baya1116/hypernet-sp-distill"
OUT = os.environ.get("RG_OUT", "results_floor_q4")
while not os.path.exists("ALL_DONE"):
    if os.path.exists("/workspace/boot.log"):
        try: api.upload_file(path_or_fileobj="/workspace/boot.log",
                             path_in_repo=f"trigger_experiment/{OUT}/boot.log", repo_id=R)
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
python build_fft_hf.py
echo "BUILD_FFT_DONE $(date -u)"

# results uploader
python - <<'PY' &
import time, os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"]); R = "baya1116/hypernet-sp-distill"
OUT = os.environ.get("RG_OUT", "results_floor_q4")
while not os.path.exists("ALL_DONE"):
    for fn in ["live.log", "results.json"]:
        p = f"{OUT}/{fn}"
        if os.path.exists(p):
            try: api.upload_file(path_or_fileobj=p, path_in_repo=f"trigger_experiment/{OUT}/{fn}", repo_id=R)
            except Exception as e: print("up_err", e, flush=True)
    time.sleep(20)
PY

echo "RUN_START $(date -u) arms=$RG_ARMS gate=$RG_GATE"
OUT_DIR="$RG_OUT" GATE_NPZ="$RG_GATE" Q4=1 GEN_N="$RG_N" TURN_CAP="$RG_TURNCAP" \
  ARMS="$RG_ARMS" python recall_gen.py
echo "RUN_DONE rc=$? $(date -u)"
touch ALL_DONE
sleep 25
python - <<'PY'
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"]); R = "baya1116/hypernet-sp-distill"
OUT = os.environ.get("RG_OUT", "results_floor_q4")
for fn in ["live.log", "results.json"]:
    p = f"{OUT}/{fn}"
    if os.path.exists(p):
        try: api.upload_file(path_or_fileobj=p, path_in_repo=f"trigger_experiment/{OUT}/{fn}", repo_id=R)
        except Exception as e: print("flush_err", e)
if os.path.exists("/workspace/boot.log"):
    try: api.upload_file(path_or_fileobj="/workspace/boot.log", path_in_repo=f"trigger_experiment/{OUT}/boot.log", repo_id=R)
    except Exception as e: print("boot_flush_err", e)
print("FLUSH_DONE")
PY
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
