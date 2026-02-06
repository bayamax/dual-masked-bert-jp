#!/bin/bash
set -e

echo "=== Phase 7: DEBUG Pipeline ==="

# 1. Pipeline Setup
echo "[1/3] Installing Dependencies..."
python3 -m pip install vllm datasets accelerate transformers peft bitsandbytes output_attentions protobuf

# 2. Data Generation (Small Scale)
echo "[2/3] Generating Small Dataset (50 samples)..."
python3 src/prep_attention_distillation.py --num_samples 50

# 3. Training (Smoke Test)
echo "[3/3] Running Training Smoke Test (1 Epoch)..."
if [ -f "phase7_attention_distill.jsonl" ]; then
    python3 src/train_phase7_distill.py --data_file phase7_attention_distill.jsonl --epochs 1 --max_steps 10
else
    echo "Error: Data file failure."
    exit 1
fi

echo "=== DEBUG Run Completed ==="
echo "Check the logs above for Loss values."
