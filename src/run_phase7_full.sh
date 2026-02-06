#!/bin/bash

# Configuration
NUM_SAMPLES=5000
EPOCHS=3

echo "=== Starting Phase 7 Full Execution ==="

# 1. Pipeline Setup: Clean previous data
# We remove the old jsonl file to start fresh (or you can keep it if you want to accumulate)
# For a clean "Large Scale Run", let's start fresh.
echo "[1/3] Clearing previous data..."
rm -f phase7_attention_distill.jsonl

# 2. Data Generation
echo "[2/3] Generating Dataset ($NUM_SAMPLES samples)..."
python3 src/prep_attention_distillation.py --num_samples $NUM_SAMPLES

# 3. Training
echo "[3/3] Running Full Training ($EPOCHS Epochs)..."
# We do NOT set --max_steps here, allowing full training
python3 src/train_phase7_distill.py --data_file phase7_attention_distill.jsonl --epochs $EPOCHS

echo "=== Phase 7 Full Execution Complete ==="
