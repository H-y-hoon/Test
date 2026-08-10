"""Train a LoRA DPO adapter from the reviewed LogicKor preference dataset."""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import math
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

from dpo_dataloader import (
    read_preference_jsonl,
    split_preference_rows,
    to_trl_preference_rows,
    validate_preference_rows,
)
from util import ensure_embedding_accessors, load_causal_lm_model, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the DPO YAML config.")
    parser.add_argument("--output-dir", required=True, help="Directory for the DPO adapter and metadata.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config, data, split, TRL row shape, and token lengths without loading a model.",
    )
    return parser.parse_args()


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping.")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer.")
    return value


def validate_config(config: dict[str, Any]) -> None:
    for key in ("model", "data_path"):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise ValueError(f"{key} must be a non-empty string.")
    split = _mapping(config.get("split"), "split")
    training = _mapping(config.get("training"), "training")
    if not 0.0 < float(split.get("train_ratio", 0.0)) < 1.0:
        raise ValueError("split.train_ratio must be between 0 and 1.")
    for key in ("max_length", "max_prompt_length", "max_completion_length"):
        _positive_int(training.get(key), f"training.{key}")
    if training["max_prompt_length"] + training["max_completion_length"] > training["max_length"]:
        raise ValueError(
            "training.max_prompt_length + max_completion_length must not exceed max_length; "
            "otherwise TRL can truncate a completion after its individual length checks."
        )
    if float(training.get("num_train_epochs", 0.0)) <= 0:
        raise ValueError("training.num_train_epochs must be positive.")
    if float(training.get("learning_rate", 0.0)) <= 0:
        raise ValueError("training.learning_rate must be positive.")
    if float(training.get("beta", 0.0)) <= 0:
        raise ValueError("training.beta must be positive.")
    for key in (
        "per_device_train_batch_size",
        "per_device_eval_batch_size",
        "gradient_accumulation_steps",
        "logging_steps",
        "eval_steps",
    ):
        _positive_int(training.get(key), f"training.{key}")
    save_checkpoints = bool(training.get("save_checkpoints", False))
    if save_checkpoints:
        save_steps = _positive_int(training.get("save_steps"), "training.save_steps")
        eval_steps = int(training["eval_steps"])
        if save_steps % eval_steps != 0:
            raise ValueError(
                "training.save_steps must be a multiple of training.eval_steps "
                "when loading the best model at the end."
            )
    if bool(training.get("load_best_model_at_end", False)) and not save_checkpoints:
        raise ValueError("training.load_best_model_at_end requires training.save_checkpoints=true.")
    if training.get("loss_type") != "sigmoid":
        raise ValueError("This LogicKor DPO entrypoint currently supports the original sigmoid DPO loss only.")
    if bool(training.get("reference_free", False)):
        raise ValueError("reference_free must remain false; the frozen SFT adapter is the DPO reference policy.")


def _completion_after_prompt(tokenizer: Any, prompt: list[dict[str, str]], answer: str) -> tuple[str, str]:
    prompt_text = tokenizer.apply_chat_template(
        prompt,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = tokenizer.apply_chat_template(
        prompt + [{"role": "assistant", "content": answer}],
        tokenize=False,
    )
    if not full_text.startswith(prompt_text):
        raise ValueError("The model chat template does not preserve the prompt prefix for a completion.")
    return prompt_text, full_text[len(prompt_text) :]


def _token_count(tokenizer: Any, text: str, append_eos: bool = False) -> int:
    count = len(tokenizer(text, add_special_tokens=False)["input_ids"])
    return count + int(append_eos and tokenizer.eos_token_id is not None)


def _distribution(values: Sequence[int]) -> dict[str, int | float]:
    ordered = sorted(values)
    percentile_50 = ordered[math.ceil(0.50 * len(ordered)) - 1]
    percentile_95 = ordered[math.ceil(0.95 * len(ordered)) - 1]
    return {
        "min": ordered[0],
        "mean": round(sum(ordered) / len(ordered), 2),
        "p50": percentile_50,
        "p95": percentile_95,
        "max": ordered[-1],
    }


def compute_token_stats(
    rows: Sequence[dict[str, Any]], tokenizer: Any, training: dict[str, Any]
) -> dict[str, Any]:
    prompt_lengths: list[int] = []
    chosen_lengths: list[int] = []
    rejected_lengths: list[int] = []
    prompt_truncated = 0
    chosen_truncated = 0
    rejected_truncated = 0
    combined_over_limit = 0
    max_prompt_length = int(training["max_prompt_length"])
    max_completion_length = int(training["max_completion_length"])
    max_length = int(training["max_length"])

    for row in rows:
        chosen_prompt, chosen_completion = _completion_after_prompt(tokenizer, row["prompt"], row["chosen"])
        rejected_prompt, rejected_completion = _completion_after_prompt(tokenizer, row["prompt"], row["rejected"])
        if chosen_prompt != rejected_prompt:
            raise RuntimeError(f"Prompt rendering differs between chosen/rejected for id {row['id']!r}.")
        prompt_length = _token_count(tokenizer, chosen_prompt)
        chosen_length = _token_count(tokenizer, chosen_completion, append_eos=True)
        rejected_length = _token_count(tokenizer, rejected_completion, append_eos=True)
        prompt_lengths.append(prompt_length)
        chosen_lengths.append(chosen_length)
        rejected_lengths.append(rejected_length)
        prompt_truncated += int(prompt_length > max_prompt_length)
        chosen_truncated += int(chosen_length > max_completion_length)
        rejected_truncated += int(rejected_length > max_completion_length)
        effective_prompt = min(prompt_length, max_prompt_length)
        effective_completion = max(
            min(chosen_length, max_completion_length),
            min(rejected_length, max_completion_length),
        )
        combined_over_limit += int(effective_prompt + effective_completion > max_length)

    return {
        "prompt_tokens": _distribution(prompt_lengths),
        "chosen_tokens": _distribution(chosen_lengths),
        "rejected_tokens": _distribution(rejected_lengths),
        "truncation_counts": {
            "prompt": prompt_truncated,
            "chosen": chosen_truncated,
            "rejected": rejected_truncated,
            "combined_after_individual_limits": combined_over_limit,
        },
    }


def prepare_data(
    config: dict[str, Any], seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    raw_rows = read_preference_jsonl(config["data_path"])
    rows, dataset_stats = validate_preference_rows(raw_rows)
    train_rows, eval_rows, split_stats = split_preference_rows(
        rows,
        train_ratio=float(config["split"]["train_ratio"]),
        seed=seed,
    )
    return train_rows, eval_rows, {"dataset_stats": dataset_stats, "split_stats": split_stats}


def normalize_trl_optional_dependency_flags() -> list[str]:
    """Work around TRL 0.24 expecting bools from a newer Transformers helper API."""
    import trl.import_utils as trl_import_utils

    normalized: list[str] = []
    for name in dir(trl_import_utils):
        if not name.startswith("_") or not name.endswith("_available"):
            continue
        value = getattr(trl_import_utils, name)
        if isinstance(value, tuple):
            setattr(trl_import_utils, name, bool(value[0]))
            normalized.append(name)
    return normalized


@contextmanager
def force_trl_text_only_preprocessing(model_type: str, enabled: bool) -> Iterator[None]:
    """Keep TRL 0.24 from demanding image fields for the text-only Gemma4 CausalLM path."""
    if not enabled:
        yield
        return
    dpo_module = importlib.import_module("trl.trainer.dpo_trainer")
    original_mapping = dpo_module.MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES
    if model_type not in original_mapping:
        yield
        return
    text_only_mapping = original_mapping.copy()
    text_only_mapping.pop(model_type, None)
    dpo_module.MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES = text_only_mapping
    try:
        yield
    finally:
        dpo_module.MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES = original_mapping


def make_dpo_config(dpo_config_cls: Any, kwargs: dict[str, Any]) -> Any:
    accepted = set(inspect.signature(dpo_config_cls).parameters)
    return dpo_config_cls(**{key: value for key, value in kwargs.items() if key in accepted})


def cleanup_best_checkpoints(output_dir: str, best_model_checkpoint: str | None) -> None:
    if not best_model_checkpoint:
        return

    best_path = Path(best_model_checkpoint).resolve()
    for checkpoint_dir in Path(output_dir).glob("checkpoint-*"):
        try:
            if checkpoint_dir.resolve() != best_path:
                shutil.rmtree(checkpoint_dir)
        except FileNotFoundError:
            pass


def make_best_checkpoint_callback(enabled: bool) -> Any | None:
    if not enabled:
        return None

    from transformers import TrainerCallback

    class BestCheckpointOnlyCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            cleanup_best_checkpoints(args.output_dir, state.best_model_checkpoint)
            return control

        def on_train_end(self, args, state, control, **kwargs):
            cleanup_best_checkpoints(args.output_dir, state.best_model_checkpoint)
            return control

    return BestCheckpointOnlyCallback()


def build_dpo_kwargs(
    output_dir: Path,
    training: dict[str, Any],
    seed: int,
    bf16: bool,
    fp16: bool,
) -> dict[str, Any]:
    save_checkpoints = bool(training.get("save_checkpoints", False))
    load_best_model_at_end = bool(training.get("load_best_model_at_end", save_checkpoints))
    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_train_epochs": float(training["num_train_epochs"]),
        "learning_rate": float(training["learning_rate"]),
        "lr_scheduler_type": str(training["lr_scheduler_type"]),
        "warmup_steps": int(training.get("warmup_steps", 0)),
        "optim": str(training["optim"]),
        "weight_decay": float(training.get("weight_decay", 0.0)),
        "per_device_train_batch_size": int(training["per_device_train_batch_size"]),
        "per_device_eval_batch_size": int(training["per_device_eval_batch_size"]),
        "gradient_accumulation_steps": int(training["gradient_accumulation_steps"]),
        "gradient_checkpointing": bool(training.get("gradient_checkpointing", True)),
        "logging_steps": int(training["logging_steps"]),
        "eval_steps": int(training["eval_steps"]),
        "eval_strategy": "steps",
        "save_strategy": "steps" if save_checkpoints else "no",
        "load_best_model_at_end": load_best_model_at_end if save_checkpoints else False,
        "metric_for_best_model": str(training.get("metric_for_best_model", "eval_loss")),
        "greater_is_better": bool(training.get("greater_is_better", False)),
        "bf16": bf16,
        "fp16": fp16,
        "seed": seed,
        "max_length": int(training["max_length"]),
        "max_prompt_length": int(training["max_prompt_length"]),
        "max_completion_length": int(training["max_completion_length"]),
        "truncation_mode": str(training.get("truncation_mode", "keep_end")),
        "beta": float(training["beta"]),
        "loss_type": str(training["loss_type"]),
        "reference_free": False,
        "use_logits_to_keep": bool(training.get("use_logits_to_keep", True)),
        "model_adapter_name": "default",
        "ref_adapter_name": "reference",
        "precompute_ref_log_probs": bool(training.get("precompute_ref_log_probs", False)),
        "dataset_num_proc": training.get("dataset_num_proc"),
        "report_to": "none",
    }
    if save_checkpoints:
        kwargs["save_steps"] = int(training["save_steps"])
        kwargs["save_total_limit"] = int(training.get("save_total_limit", 1))
    return kwargs


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def run(args: argparse.Namespace, config: dict[str, Any]) -> None:
    validate_config(config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_rows, eval_rows, data_meta = prepare_data(config, args.seed)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config["model"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not tokenizer.chat_template:
        raise ValueError(f"The DPO starting checkpoint has no chat template: {config['model']}")
    trl_train_rows = to_trl_preference_rows(train_rows)
    trl_eval_rows = to_trl_preference_rows(eval_rows)
    if not all(isinstance(row["chosen"], list) and isinstance(row["rejected"], list) for row in trl_train_rows + trl_eval_rows):
        raise RuntimeError("TRL preference conversion produced a non-conversational completion.")
    token_stats = compute_token_stats(train_rows + eval_rows, tokenizer, config["training"])
    base_meta = {
        "seed": args.seed,
        "model": config["model"],
        "data_path": config["data_path"],
        "base_model": base_model_name if "base_model_name" in locals() else None,
        **data_meta,
        "token_stats": token_stats,
        "trl_contract": "prompt messages + assistant-message chosen/rejected",
    }
    if args.dry_run:
        payload = {"mode": "dry_run", **base_meta}
        write_json(output_dir / "dry_run_meta.json", payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    training = config["training"]
    cuda_visible_devices = training.get("cuda_visible_devices")
    if cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    import torch
    from datasets import Dataset
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, set_seed

    normalized_flags = normalize_trl_optional_dependency_flags()
    from trl import DPOConfig, DPOTrainer

    set_seed(args.seed)
    has_cuda = bool(torch.cuda.is_available())
    bf16 = bool(has_cuda and torch.cuda.is_bf16_supported() and training.get("bf16", True))
    fp16 = bool(has_cuda and not bf16 and training.get("fp16", False))
    model_dtype = torch.bfloat16 if bf16 else torch.float16 if has_cuda else torch.float32
    sft_adapter_config = PeftConfig.from_pretrained(config["model"])
    base_model_name = sft_adapter_config.base_model_name_or_path
    base_model = load_causal_lm_model(AutoModelForCausalLM, base_model_name, model_dtype)
    base_model.config.use_cache = False
    ensure_embedding_accessors(base_model)
    model = PeftModel.from_pretrained(
        base_model,
        config["model"],
        adapter_name="default",
        is_trainable=True,
    )
    model.load_adapter(
        config["model"],
        adapter_name="reference",
        is_trainable=False,
    )
    model.set_adapter("default")
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}
    dpo_kwargs = build_dpo_kwargs(output_dir, training, args.seed, bf16, fp16)
    dpo_args = make_dpo_config(DPOConfig, dpo_kwargs)
    train_dataset = Dataset.from_list(trl_train_rows)
    eval_dataset = Dataset.from_list(trl_eval_rows)
    with force_trl_text_only_preprocessing(
        model.config.model_type,
        enabled=bool(config.get("force_text_only", False)),
    ):
        callbacks = []
        best_checkpoint_callback = make_best_checkpoint_callback(
            bool(training.get("best_checkpoint_only", False))
        )
        if best_checkpoint_callback is not None:
            callbacks.append(best_checkpoint_callback)
        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=dpo_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            peft_config=None,
            callbacks=callbacks,
        )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(adapter_dir, selected_adapters=["default"])
    tokenizer.save_pretrained(adapter_dir)

    run_meta = {
        **base_meta,
        "base_model": base_model_name,
        "train_args": dpo_kwargs,
        "trl_optional_flags_normalized": normalized_flags,
        "reference_policy": "frozen copy of the starting SFT adapter on the shared base model",
        "global_step": trainer.state.global_step,
        "best_checkpoint_only": bool(training.get("best_checkpoint_only", False)),
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "best_eval_loss": trainer.state.best_metric,
        "adapter_dir": str(adapter_dir),
    }
    write_json(output_dir / "run_meta.json", run_meta)
    print(json.dumps(run_meta, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    run(args, load_config(args.config))


if __name__ == "__main__":
    main()
