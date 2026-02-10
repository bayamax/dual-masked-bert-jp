#!/bin/bash
set -e

echo "=== Starting Phase 10 Data Preparation ==="

# 1. Generate GSM8K (Math) - Full set (~7.5k)
echo "[1/3] Generating GSM8K Data..."
./venv/bin/python3 src/gen_phase9_math.py --num_samples 10000

# 2. Generate Alpaca (General) - Limited to 10k for Composite Training
echo "[2/3] Generating Alpaca Data (Limit 10k)..."
./venv/bin/python3 src/gen_phase8_alpaca.py --num_samples 10000

# 3. Mix Data
echo "[3/3] Creating Composite Dataset..."
./venv/bin/python3 src/prepare_phase10_data.py

echo "=== Phase 10 Data Ready: phase10_composite.jsonl ==="
ls -lh phase10_composite.jsonl
