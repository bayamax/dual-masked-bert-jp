#!/bin/bash
set -e

# Configuration
NUM_SMOKE_SAMPLES=10
EPOCHS=1
MAX_STEPS=5

echo "=== Starting Phase 7 vLLM Smoke Test ==="

# 1. Clean previous data
echo "[1/4] Clearing previous data..."
rm -f phase7_raw_cot.jsonl
rm -f phase7_attention_distill.jsonl

# 2. vLLM Generation (Fast)
echo "[2/4] Generating Raw CoT with vLLM ($NUM_SMOKE_SAMPLES samples)..."
python3 src/gen_phase7_vllm.py --num_samples $NUM_SMOKE_SAMPLES

# 3. Attention Processing (Batch)
echo "[3/4] Processing Attention Weights..."
python3 src/process_phase7_attention.py

# 4. Training Smoke Test
echo "[4/4] Running Training Smoke Test..."
python3 src/train_phase7_distill.py --data_file phase7_attention_distill.jsonl --epochs $EPOCHS --max_steps $MAX_STEPS

echo "=== Phase 7 vLLM Smoke Test Complete ==="
