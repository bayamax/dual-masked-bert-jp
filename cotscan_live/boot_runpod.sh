#!/bin/bash
# RunPod boot: reconstruct the faithful runtime from PUBLIC HF + apply our patch, run a
# conversation battery on GPU, and serve the results over HTTP :8000 (RunPod proxy).
exec > /workspace/boot.log 2>&1
set -x
export HF_HUB_ENABLE_HF_TRANSFER=0 SPCHAT_DEVICE=cuda PYTHONUNBUFFERED=1
cd /workspace
pip install -q "transformers>=4.46,<4.54" "datasets>=2.18" sentence-transformers joblib "huggingface_hub>=0.25" requests beautifulsoup4 certifi lxml 2>&1 | tail -2
B="https://raw.githubusercontent.com/bayamax/dual-masked-bert-jp/claude/hypernet-sp-distill-d3pyik/cotscan_live"
mkdir -p run && cd run
for f in faithful_setup.py faithful_fixes.py conv_web.py; do curl -fsSL "$B/$f" -o "$f"; done
mkdir -p faithful_patches && curl -fsSL "$B/faithful_patches/app_session_torch.patch" -o faithful_patches/app_session_torch.patch
echo "STEP setup"; python faithful_setup.py
echo "STEP build"; ( cd faithful_run && python build_fft_hf.py )
echo "STEP patch"; python faithful_fixes.py
echo "STEP conv6"; python conv_web.py 2>&1 | tee conv_web.console
cp faithful_run/conv_web.log conv_web.log 2>/dev/null || true
cp /workspace/boot.log boot.log
echo "STEP serve"; echo DONE > .done
python -m http.server 8000
