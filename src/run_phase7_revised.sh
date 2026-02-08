#!/bin/bash
set -e

echo "=== Phase 7 Revised: CoT Compression Training Pipeline ==="

# 1. Prepare chunks
echo "[1/2] Preparing CoT chunks..."
python3 src/prepare_phase7_cot_chunks.py --max_samples 1000

# 2. Train
echo "[2/2] Training CoT continuation model..."
python3 src/train_phase7_cot_continuation.py --epochs 3

echo "=== Pipeline Complete ==="
