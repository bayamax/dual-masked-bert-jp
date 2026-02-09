#!/bin/bash
set -e

echo "Resuming Phase 9 Pipeline (Data Gen Complete)"
date

# 1. Skip Gen (File exists: 7473 lines)

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

echo "Phase 9 Manual Resume Completed Successfully!"
date
