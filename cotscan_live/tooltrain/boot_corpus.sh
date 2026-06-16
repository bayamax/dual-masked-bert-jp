#!/bin/bash
exec > /workspace/boot.log 2>&1
set -x
export HF_HUB_ENABLE_HF_TRANSFER=0 PYTHONUNBUFFERED=1
pip install -q "transformers>=4.46,<4.54" "datasets>=2.18" scikit-learn "huggingface_hub>=0.25" 2>&1 | tail -2
cd /workspace
python - <<'PY'
from huggingface_hub import hf_hub_download
import shutil, os
R="baya1116/hypernet-sp-distill"; os.makedirs("fft_out",exist_ok=True)
for p in ["build_fft_hf.py","fft_out/student.pt"]: shutil.copy(hf_hub_download(R,p), p)
print("weights ok")
PY
python build_fft_hf.py 2>&1 | tail -1
B="https://raw.githubusercontent.com/bayamax/dual-masked-bert-jp/claude/hypernet-sp-distill-d3pyik/cotscan_live/tooltrain"
curl -fsSL "$B/build_corpus.py" -o build_corpus.py
echo "RUN_CORPUS $(date -u)"
python build_corpus.py 2>&1 | tee corpus_run.log
echo "CORPUS_DONE $(date -u)"
python -m http.server 8000
