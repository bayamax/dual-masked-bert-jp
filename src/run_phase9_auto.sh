#!/bin/bash
set -e

echo "Starting Phase 9 Auto Pipeline..."
date

# 1. Data Generation (if not already done/running)
echo ">> Step 1: Data Generation (GSM8K)"
python3 src/gen_phase9_math.py
echo "Generation Finished. Waiting 30s for cleanup..."
sleep 30

# 2. Data Preparation
echo ">> Step 2: Data Preparation"
python3 src/prepare_phase9_math.py
echo "Preparation Finished. Waiting 30s..."
sleep 30

# 3. Training
echo ">> Step 3: Training"
python3 src/train_phase9_math.py
echo "Training Finished. Waiting 30s..."
sleep 30

# 4. Verification
echo ">> Step 4: Verification"
python3 src/verify_phase9_math.py

echo "Phase 9 Pipeline Completed Successfully!"
date
