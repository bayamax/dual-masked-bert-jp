#!/bin/bash
# Optimized Training Script
# Batch Size increased to 16 based on hardware capacity (RTX 4090/24GB)

BATCH_SIZE=16
EPOCHS=3

echo "=== Starting Phase 7 Training (Optimized BS=$BATCH_SIZE) ==="
python3 src/train_phase7_distill.py --data_file phase7_attention_distill.jsonl --epochs $EPOCHS --batch_size $BATCH_SIZE
