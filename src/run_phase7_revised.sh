#!/bin/bash
set -e

echo "=== Phase 7 Revised: CoT Compression Training Pipeline ==="

# 1. Prepare chunks (5000 samples for sufficient data)
echo "[1/2] Preparing CoT chunks (5000 samples)..."
python3 src/prepare_phase7_cot_chunks.py --max_samples 5000

# 2. Train (1 epoch to avoid overfitting, batch_size=8 for speed)
echo "[2/2] Training CoT continuation model (1 epoch)..."
python3 src/train_phase7_cot_continuation.py --epochs 1

echo "=== Pipeline Complete ==="

echo "=== Pipeline Complete ==="
