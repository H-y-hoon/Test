#!/usr/bin/env bash
set -euo pipefail

# Experiment parameters. Edit these values before running the script.
TASKS="data/logickor_sft_high_v2.jsonl"
RUN_DIR="data/logickor_sft_candidate_v2_full"
LORA_PATH="runs/gemma4_e4b_step_sft_v2/adapter"

MODEL="google/gemma-4-E4B-it"
MAX_LORA_RANK=32

GPU_0=2
GPU_1=3
SEED_0=42
SEED_1=1042

CANDIDATES_PER_GPU=2
REQUIRED_CANDIDATES=4
BATCH_SIZE=8
DTYPE="bfloat16"
MAX_MODEL_LEN=4096
MAX_TOKENS=1360
TEMPERATURE=0.7
TOP_P=0.9

mkdir -p "${RUN_DIR}"

python augmentation/generate_sft_candidates_vllm.py generate \
  --tasks "${TASKS}" \
  --output "${RUN_DIR}/worker_0_candidates.jsonl" \
  --failures "${RUN_DIR}/worker_0_failures.jsonl" \
  --model "${MODEL}" \
  --lora-path "${LORA_PATH}" \
  --max-lora-rank "${MAX_LORA_RANK}" \
  --gpu-device "${GPU_0}" \
  --worker-id worker_0 \
  --candidate-count "${CANDIDATES_PER_GPU}" \
  --batch-size "${BATCH_SIZE}" \
  --dtype "${DTYPE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-tokens "${MAX_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --top-p "${TOP_P}" \
  --seed "${SEED_0}" \
  --enforce-eager \
  --disable-weight-tracking \
  --language-model-only &
PID_0=$!

python augmentation/generate_sft_candidates_vllm.py generate \
  --tasks "${TASKS}" \
  --output "${RUN_DIR}/worker_1_candidates.jsonl" \
  --failures "${RUN_DIR}/worker_1_failures.jsonl" \
  --model "${MODEL}" \
  --lora-path "${LORA_PATH}" \
  --max-lora-rank "${MAX_LORA_RANK}" \
  --gpu-device "${GPU_1}" \
  --worker-id worker_1 \
  --candidate-count "${CANDIDATES_PER_GPU}" \
  --batch-size "${BATCH_SIZE}" \
  --dtype "${DTYPE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-tokens "${MAX_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --top-p "${TOP_P}" \
  --seed "${SEED_1}" \
  --enforce-eager \
  --disable-weight-tracking \
  --language-model-only &
PID_1=$!

wait "${PID_0}"
wait "${PID_1}"

python augmentation/generate_sft_candidates_vllm.py merge \
  --tasks "${TASKS}" \
  --worker-output "${RUN_DIR}/worker_0_candidates.jsonl" \
  --worker-output "${RUN_DIR}/worker_1_candidates.jsonl" \
  --required-candidates "${REQUIRED_CANDIDATES}" \
  --output "${RUN_DIR}/merged_candidates.jsonl" \
  --incomplete "${RUN_DIR}/incomplete_ids.jsonl"
