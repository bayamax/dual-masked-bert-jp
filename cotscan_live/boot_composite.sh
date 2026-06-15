#!/bin/bash
set +e
exec > /workspace/boot.log 2>&1
echo "BOOT_START $(date -u)"
export HF_HUB_ENABLE_HF_TRANSFER=0
GH="https://raw.githubusercontent.com/bayamax/dual-masked-bert-jp/e664ec9bb30b5a9a536439f0f87f498d887e2a31/trigger_experiment"
O="trigger_experiment/recall_kit"

pip install -q -U "huggingface_hub>=0.25" "transformers>=4.46,<4.54" "datasets>=2.18" "scikit-learn>=1.3" "sentence-transformers>=3.0" requests 2>&1 | tail -2
echo "PIP_DONE $(date -u)"
mkdir -p /work && cd /work && mkdir -p recall_kit_artifacts

python - <<'PY'
import os, shutil
from huggingface_hub import hf_hub_download
R="baya1116/hypernet-sp-distill"; tok=os.environ["HF_TOKEN"]
for p in ["hypernet_sp/attn_export3_torch.py","hypernet_sp/attn_scenarios.py",
          "hypernet_sp/block_recall.py","build_fft_hf.py","fft_out/student.pt","fft_out/pooler.pt"]:
    f=hf_hub_download(R,p,token=tok); dst=p if p.startswith("fft_out/") else os.path.basename(p)
    os.makedirs(os.path.dirname(dst) or ".",exist_ok=True); shutil.copy(f,dst)
# reuse the already-trained component weights
for w in ["gate.npz","indexer.npz","bge_head.npz"]:
    f=hf_hub_download(R,f"trigger_experiment/recall_kit/artifacts/{w}",token=tok)
    shutil.copy(f,f"recall_kit_artifacts/{w}"); print("got",w)
print("HF_DONE")
PY
python - <<PY
import urllib.request
open("cot_recall_eval.py","wb").write(urllib.request.urlopen("$GH/cot_recall_eval.py",timeout=60).read())
PY
sed -i 's/, dtype=torch\.float32/, torch_dtype=torch.float32/g' cot_recall_eval.py 2>/dev/null

echo "$DOLPHIN_PY_B64"  | base64 -d > dolphin_scan.py
echo "$COMPOSITE_PY_B64"| base64 -d > composite_test.py
echo "$RECALL_KIT_B64"  | base64 -d > recall_kit.tgz && tar xzf recall_kit.tgz
python -c "import recall_kit; print('recall_kit OK')"
echo "BUILD_FFT $(date -u)"; python build_fft_hf.py 2>&1 | tail -1; echo "BUILD_FFT_DONE $(date -u)"

python - <<'PY' &
import time, os
from huggingface_hub import HfApi
api=HfApi(token=os.environ["HF_TOKEN"]); R="baya1116/hypernet-sp-distill"; O="trigger_experiment/recall_kit"
while True:
    for fn in ["composite.log","composite_report.json"]:
        p=f"recall_kit_artifacts/{fn}"
        if os.path.exists(p):
            try: api.upload_file(path_or_fileobj=p,path_in_repo=f"{O}/artifacts/{fn}",repo_id=R)
            except Exception as e: print("up",e,flush=True)
    if os.path.exists("DONE"): break
    time.sleep(20)
PY

echo "COMPOSITE $(date -u)"
ART_DIR=recall_kit_artifacts OUT_DIR=recall_kit_artifacts POS_FRAC=0.08 python -u composite_test.py
echo "COMPOSITE_DONE rc=$? $(date -u)"
# also re-upload the fixed runtime source
python - <<'PY'
import os, glob
from huggingface_hub import HfApi
api=HfApi(token=os.environ["HF_TOKEN"]); R="baya1116/hypernet-sp-distill"; O="trigger_experiment/recall_kit"
for f in glob.glob("recall_kit/*.py"):
    api.upload_file(path_or_fileobj=f,path_in_repo=f"{O}/src/{os.path.basename(f)}",repo_id=R)
for fn in ["composite_report.json","composite.log"]:
    p=f"recall_kit_artifacts/{fn}"
    if os.path.exists(p): api.upload_file(path_or_fileobj=p,path_in_repo=f"{O}/artifacts/{fn}",repo_id=R)
print("UPLOAD_DONE")
PY
cp /workspace/boot.log recall_kit_artifacts/composite_boot.log
touch DONE; sleep 20
python - <<'PY'
import os, requests
pid=os.environ.get("RUNPOD_POD_ID");key=os.environ.get("RUNPOD_API_KEY")
if pid and key: requests.post("https://api.runpod.io/graphql",headers={"Authorization":f"Bearer {key}","Content-Type":"application/json"},json={"query":'mutation{podTerminate(input:{podId:"%s"})}'%pid},timeout=30); print("TERM")
PY
echo "BOOT_END $(date -u)"
