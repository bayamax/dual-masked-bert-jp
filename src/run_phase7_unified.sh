#!/bin/bash
set -e

echo "=== Phase 7 Unified Training Pipeline ==="

# Training
echo "[1/2] Running Unified Training (3 epochs)..."
python3 src/train_phase7_unified.py --epochs 3

# Verification
echo "[2/2] Running Verification..."
python3 src/verify_phase7_unified.py

echo "=== Phase 7 Unified Pipeline Complete ==="
