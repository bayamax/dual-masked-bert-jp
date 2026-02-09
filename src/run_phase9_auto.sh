#!/bin/bash
set -e

echo "Starting Phase 9 Auto Pipeline..."
date

# 1. Data Generation (if not already done/running)
# Note: The user started this manually. If it's running, we should wait or just rely on the file?
# To be safe, we can run it again with a check, or just skip if file exists?
# The python script supports resuming.
echo ">> Step 1: Data Generation (GSM8K)"
python3 src/gen_phase9_math.py

# 2. Data Preparation
echo ">> Step 2: Data Preparation"
python3 src/prepare_phase9_math.py

# 3. Training
echo ">> Step 3: Training"
python3 src/train_phase9_math.py

# 4. Verification
echo ">> Step 4: Verification"
python3 src/verify_phase9_math.py

echo "Phase 9 Pipeline Completed Successfully!"
date
