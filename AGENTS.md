# Repository Guidelines

## Project Structure & Module Organization

This repository is a minimal LogicKor SFT reproduction package. Training code lives in `train/`; LogicKor generation, judge evaluation, and scoring live in `logickor_eval/`; preference-data augmentation code lives in `augmentation/`. Training configs are in `configs/`, runnable shell examples are in `scripts/`, dependency pins are in `requirements/`, and augmentation materials are in `prompts/`. Large local artifacts belong in `data/`, `models/`, `runs/`, `generated/`, `evaluated/`, and `results/`; these directories are gitignored except for `.gitkeep`.

## Build, Test, and Development Commands

There is no package build step. Use Python 3.12 and the split environments from the README:

```bash
conda create -n etri python=3.12 -y
pip install -r requirements/etri-training.txt

conda create -n etri-infer python=3.12 -y
pip install -r requirements/etri-infer.txt
```

Common commands:

```bash
bash scripts/train.sh configs/train_qwen3_8b_sft.yaml runs/qwen3_8b_sft_high
bash scripts/generate.sh models/qwen3_8b_sft_high/merged
bash scripts/evaluate.sh generated/models/qwen3_8b_sft_high/merged
bash scripts/score.sh 'evaluated/models/qwen3_8b_sft_high/merged/*.jsonl'
python train/train_lora.py --config configs/train_qwen3_8b_sft.yaml --output-dir /tmp/logickor-dry-run --dry-run
```

`evaluate.sh` requires `OPENAI_API_KEY` in the environment.

## Coding Style & Naming Conventions

Use 4-space indentation, type hints for new helpers, and `Path`/stdlib utilities before adding dependencies. Keep modules script-friendly with `argparse` for command-line entrypoints. Use descriptive snake_case for Python names and lowercase, model-specific filenames for configs, for example `train_qwen3_8b_sft.yaml`.

## Testing Guidelines

No formal test suite is present. Before submitting Python changes, run at least:

```bash
python -m py_compile train/*.py logickor_eval/*.py
python train/train_lora.py --config configs/train_qwen3_8b_sft.yaml --output-dir /tmp/logickor-dry-run --dry-run
```

For evaluation changes, run generation/evaluation/scoring on a small local model or saved output and record the command used.

## Commit & Pull Request Guidelines

Recent commits use short imperative summaries such as `Prepare Qwen3-8B LogicKor SFT reproduction package` and `Document why the shared SFT model was selected`. Keep subjects concise and explain data/model assumptions in the body when relevant. Pull requests should include purpose, changed commands or configs, required local artifacts, verification output, and linked issues. Do not commit API keys, `.env` files, downloaded datasets, model weights, or generated run outputs.
