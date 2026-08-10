#!/usr/bin/env bash
set -euo pipefail

python train/train_dpo.py \
  --config configs/train_gemma4_e4b_dpo.yaml \
  --output-dir runs/gemma4_e4b_dpo_v3 \
  --seed 42
