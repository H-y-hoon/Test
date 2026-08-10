#!/usr/bin/env bash
set -euo pipefail

# Experiment parameters. Edit these values before running the script.
INPUT="data/logickor_sft_candidate_full/merged_candidates.jsonl"
RUN_DIR="data/logickor_sft_candidate_judge_full"
MODEL="google/gemma-4-31B-it"
PROMPT_DIR="prompts/judge"

GPU_DEVICES="2,3"
TENSOR_PARALLEL_SIZE=2
BATCH_SIZE=2
GROUP_SIZE=2
DTYPE="bfloat16"
MAX_MODEL_LEN=4096
MAX_TOKENS=256
TEMPERATURE=0
SEED=42
GPU_MEMORY_UTILIZATION=0.95
CPU_OFFLOAD_GB=2

mkdir -p "${RUN_DIR}"

python augmentation/judge_sft_candidates_vllm.py \
  --input "${INPUT}" \
  --pair-output "${RUN_DIR}/pairwise_judgments.jsonl" \
  --output "${RUN_DIR}/judged_candidates.jsonl" \
  --failures "${RUN_DIR}/failures.jsonl" \
  --incomplete "${RUN_DIR}/incomplete_ids.jsonl" \
  --prompt-dir "${PROMPT_DIR}" \
  --model "${MODEL}" \
  --gpu-devices "${GPU_DEVICES}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --batch-size "${BATCH_SIZE}" \
  --group-size "${GROUP_SIZE}" \
  --dtype "${DTYPE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-tokens "${MAX_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --seed "${SEED}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --cpu-offload-gb "${CPU_OFFLOAD_GB}"
