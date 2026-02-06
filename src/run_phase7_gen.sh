#!/bin/bash
# Phase 7 Data Generation Launcher

echo "Setup: Installing vllm and datasets..."
python3 -m pip install vllm datasets accelerate --upgrade

echo "Starting Data Generation with DeepSeek-R1 (Attention Distillation)..."
python3 src/prep_attention_distillation.py

echo "Done! Check phase7_sft_data.jsonl"
