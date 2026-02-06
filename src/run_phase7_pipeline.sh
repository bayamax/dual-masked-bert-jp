#!/bin/bash
set -e

echo "=== Phase 7: Setup & Execution Pipeline ==="

# 1. Install Dependencies
echo "[1/3] Installing Dependencies..."
python3 -m pip install -U pip
python3 -m pip install vllm datasets accelerate transformers peft bitsandbytes output_attentions

# 2. Data Generation
echo "[2/3] Starting Attention Distillation Data Generation..."
python3 src/prep_attention_distillation.py

# 3. Training
echo "[3/3] Starting HyperNet + LoRA Distillation Training..."
# Ensure the data file exists
if [ -f "phase7_attention_distill.jsonl" ]; then
    python3 src/train_phase7_distill.py --data_file phase7_attention_distill.jsonl
else
    echo "Error: Data file phase7_attention_distill.jsonl not found!"
    exit 1
fi

echo "=== Pipeline Completed Successfully ==="
