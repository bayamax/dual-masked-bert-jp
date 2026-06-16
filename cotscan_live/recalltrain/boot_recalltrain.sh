#!/bin/bash
exec > /workspace/boot.log 2>&1
set -x
export HF_HUB_ENABLE_HF_TRANSFER=0 PYTHONUNBUFFERED=1
pip install -q "transformers>=4.46,<4.54" "peft>=0.11" "datasets>=2.18" "huggingface_hub>=0.25" 2>&1 | tail -1
cd /workspace
python - <<'PY'
from huggingface_hub import hf_hub_download
import shutil, os
R="baya1116/hypernet-sp-distill"; os.makedirs("fft_out",exist_ok=True)
for p in ["fft_out/student.pt","fft_out/pooler.pt"]: shutil.copy(hf_hub_download(R,p), p)
for p in ["build_fft_hf.py","hypernet_sp/block_recall.py","hypernet_sp/attn_export3_torch.py","hypernet_sp/attn_scenarios.py"]:
    shutil.copy(hf_hub_download(R,p), os.path.basename(p))
print("ok", os.listdir("fft_out"))
PY
python build_fft_hf.py 2>&1 | tail -2
ls -d fft_hf >/dev/null 2>&1 && echo FFT_HF_OK || echo FFT_HF_MISSING
B="https://raw.githubusercontent.com/bayamax/dual-masked-bert-jp/claude/hypernet-sp-distill-d3pyik/cotscan_live/recalltrain"
curl -fsSL "$B/recall_sft.py" -o recall_sft.py
curl -fsSL "$B/recall_eval.py" -o recall_eval.py
echo "TRAIN $(date -u)"; N=${N:-400} EP=${EP:-1} python recall_sft.py 2>&1 | tee train.log
echo "EVAL $(date -u)"; python recall_eval.py 2>&1 | tee eval.log
echo "ALLDONE $(date -u)"; python -m http.server 8000
