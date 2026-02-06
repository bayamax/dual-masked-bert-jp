#!/bin/bash
set -e

echo "=== Phase 7 SFT Training Pipeline ==="

# 1. Prepare SFT Data
echo "[1/2] Preparing SFT dataset..."
python3 src/prepare_phase7_sft_injection.py

# 2. Train
echo "[2/2] Training SFT model (3 epochs)..."
python3 src/train_phase7_sft_injection.py --epochs 3

echo "=== Phase 7 SFT Complete ==="
