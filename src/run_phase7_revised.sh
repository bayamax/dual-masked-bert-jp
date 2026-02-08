#!/bin/bash
set -e

echo "=== Phase 7 Revised: CoT Compression Training Pipeline ==="

# 1. Prepare chunks (10000 samples for accuracy boost)
echo "[1/2] Preparing CoT chunks (10000 samples)..."
python3 src/prepare_phase7_cot_chunks.py --max_samples 10000

# 2. Train (Resume from Epoch 0, Train 5 Epochs with Validation)
echo "[2/2] Training CoT continuation model (Validation + Best Save)..."
python3 src/train_phase7_cot_continuation.py \
    --epochs 5 \
    --resume_lora phase7_revised_lora_epoch0 \
    --resume_hypernet phase7_revised_hypernet_epoch0.pt \
    --output_dir phase7_accuracy_boost

echo "=== Pipeline Complete ==="

echo "=== Pipeline Complete ==="
