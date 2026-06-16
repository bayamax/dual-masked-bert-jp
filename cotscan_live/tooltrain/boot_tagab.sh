#!/bin/bash
exec > /workspace/boot.log 2>&1
set -x
export HF_HUB_ENABLE_HF_TRANSFER=0 PYTHONUNBUFFERED=1
pip install -q "transformers>=4.46,<4.54" "huggingface_hub>=0.25" 2>&1 | tail -1
cd /workspace
python - <<'PY'
from huggingface_hub import hf_hub_download
import shutil, os
R="baya1116/hypernet-sp-distill"; os.makedirs("fft_out",exist_ok=True)
for p in ["build_fft_hf.py","fft_out/student.pt","fft_out/pooler.pt","hypernet_sp/block_recall.py","hypernet_sp/attn_export3_torch.py","hypernet_sp/attn_scenarios.py"]:
    shutil.copy(hf_hub_download(R,p), os.path.basename(p))
print("ok")
PY
python build_fft_hf.py 2>&1 | tail -1
B="https://raw.githubusercontent.com/bayamax/dual-masked-bert-jp/claude/hypernet-sp-distill-d3pyik/cotscan_live/tooltrain"
curl -fsSL "$B/recall_tag_ab.py" -o recall_tag_ab.py
echo "RUN $(date -u)"; python recall_tag_ab.py 2>&1 | tee tagab_run.log
echo "DONE $(date -u)"; python -m http.server 8000
