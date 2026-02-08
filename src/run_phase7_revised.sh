#!/bin/bash
set -e

echo "=== Phase 7 Revised: CoT Compression Training Pipeline ==="

# 1. Prepare chunks (5000 samples for sufficient data)
echo "[1/2] Preparing CoT chunks (5000 samples)..."
python3 src/prepare_phase7_cot_chunks.py --max_samples 5000

# 2. Train (Resume from Epoch 0, Train 1 more Epoch with Retrieval Fix)
echo "[2/2] Training CoT continuation model (Retrieval Fix + DeepSeek Format)..."
python3 src/train_phase7_cot_continuation.py \
    --epochs 1 \
    --resume_lora phase7_revised_lora_epoch0 \
    --resume_hypernet phase7_revised_hypernet_epoch0.pt \
    --output_dir phase7_retrieval_fix_epoch0

echo "=== Pipeline Complete ==="

echo "=== Pipeline Complete ==="
