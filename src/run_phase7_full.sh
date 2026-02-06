#!/bin/bash
set -e

# Configuration
NUM_SAMPLES=5000
EPOCHS=3

echo "=== Starting Phase 7 Full Execution ==="

# 1. Pipeline Setup: Clean previous data
# We remove the old jsonl file to start fresh (or you can keep it if you want to accumulate)
# For a clean "Large Scale Run", let's start fresh.
echo "[1/3] Clearing previous data..."
rm -f phase7_attention_distill.jsonl

# 2. Data Generation (Optimized vLLM pipeline)
echo "[2/4] Generating Raw CoT with vLLM ($NUM_SAMPLES samples)..."
python3 src/gen_phase7_vllm.py --num_samples $NUM_SAMPLES

# 3. Attention Processing
echo "[3/4] Processing Attention Weights..."
python3 src/process_phase7_attention.py

# 4. Training
echo "[4/4] Running Full Training ($EPOCHS Epochs)..."
# We do NOT set --max_steps here, allowing full training
python3 src/train_phase7_distill.py --data_file phase7_attention_distill.jsonl --epochs $EPOCHS

echo "=== Phase 7 Full Execution Complete ==="
