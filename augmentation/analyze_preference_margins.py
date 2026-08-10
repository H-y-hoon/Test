"""Measure how the starting SFT policy ranks reviewed chosen/rejected pairs.

The primary diagnostic is the difference between the mean completion-token
log-probability of chosen and rejected.  Summed log-probabilities are also
reported, but they are strongly affected by answer length and should not be
used alone to filter preference data.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.dpo_dataloader import (  # noqa: E402
    read_preference_jsonl,
    split_preference_rows,
    validate_preference_rows,
)
from train.util import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="DPO YAML config used for the experiment.")
    parser.add_argument(
        "--model",
        help="Optional starting SFT adapter override; data and length settings still come from config.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--gpu-device",
        help="Physical CUDA device. Defaults to training.cuda_visible_devices in the config.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Reproduce the DPO train/eval split.")
    parser.add_argument(
        "--near-zero-threshold",
        type=float,
        default=0.05,
        help="Heuristic uncertainty band in mean log-probability nats per completion token.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress after this many pairs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional pilot limit. Omit to process the full dataset.",
    )
    parser.add_argument(
        "--use-logits-to-keep",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Ask supported models to return only completion-adjacent logits.",
    )
    return parser.parse_args()


def percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def distribution(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "min": min(values),
        "p10": percentile(values, 0.10),
        "p25": percentile(values, 0.25),
        "p50": percentile(values, 0.50),
        "p75": percentile(values, 0.75),
        "p90": percentile(values, 0.90),
        "max": max(values),
        "mean": statistics.fmean(values),
        "stdev": statistics.pstdev(values),
    }


def pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right, strict=True))
    left_scale = math.sqrt(sum((x - left_mean) ** 2 for x in left))
    right_scale = math.sqrt(sum((y - right_mean) ** 2 for y in right))
    if left_scale == 0.0 or right_scale == 0.0:
        return None
    return numerator / (left_scale * right_scale)


def preference_counts(rows: Sequence[dict[str, Any]], threshold: float) -> dict[str, int]:
    margins = [float(row["mean_margin"]) for row in rows]
    return {
        "chosen_above_band": sum(value > threshold for value in margins),
        "near_zero_band": sum(abs(value) <= threshold for value in margins),
        "rejected_above_band": sum(value < -threshold for value in margins),
        "chosen_positive": sum(value > 0.0 for value in margins),
        "rejected_positive": sum(value < 0.0 for value in margins),
        "exact_tie": sum(value == 0.0 for value in margins),
    }


def summarize_rows(rows: Sequence[dict[str, Any]], threshold: float) -> dict[str, Any]:
    if not rows:
        return {"count": 0}
    length_deltas = [float(row["rejected_tokens"] - row["chosen_tokens"]) for row in rows]
    sum_margins = [float(row["sum_margin"]) for row in rows]
    mean_margins = [float(row["mean_margin"]) for row in rows]
    return {
        "count": len(rows),
        "preference_counts": preference_counts(rows, threshold),
        "sum_margin_sign_counts": {
            "chosen_positive": sum(value > 0.0 for value in sum_margins),
            "rejected_positive": sum(value < 0.0 for value in sum_margins),
            "exact_tie": sum(value == 0.0 for value in sum_margins),
        },
        "sum_margin": distribution(sum_margins),
        "mean_margin": distribution(mean_margins),
        "chosen_tokens": distribution([float(row["chosen_tokens"]) for row in rows]),
        "rejected_tokens": distribution([float(row["rejected_tokens"]) for row in rows]),
        "rejected_minus_chosen_tokens": distribution(length_deltas),
        "correlation_length_delta_sum_margin": pearson(length_deltas, sum_margins),
        "correlation_length_delta_mean_margin": pearson(length_deltas, mean_margins),
    }


def grouped_summary(
    rows: Sequence[dict[str, Any]], key: str, threshold: float
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key, "unknown"))].append(row)
    return {
        group: summarize_rows(group_rows, threshold)
        for group, group_rows in sorted(groups.items())
    }


def render_token_ids(
    tokenizer: Any,
    prompt: list[dict[str, str]],
    answer: str,
    max_prompt_length: int,
    max_completion_length: int,
    max_length: int,
) -> tuple[list[int], int, dict[str, Any]]:
    prompt_text = tokenizer.apply_chat_template(
        prompt,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = tokenizer.apply_chat_template(
        prompt + [{"role": "assistant", "content": answer}],
        tokenize=False,
        add_generation_prompt=False,
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

    prompt_length = len(prompt_ids)
    if full_ids[:prompt_length] != prompt_ids:
        # Tokenization can merge one boundary token when the completion is appended.
        if prompt_length < 1 or full_ids[: prompt_length - 1] != prompt_ids[:-1]:
            raise ValueError("Full conversation tokenization does not preserve the prompt prefix.")
        prompt_length -= 1

    effective_prompt = list(full_ids[:prompt_length])
    completion_ids = list(full_ids[prompt_length:])
    eos_token_id = tokenizer.eos_token_id
    eos_appended = eos_token_id is not None and (not completion_ids or completion_ids[-1] != eos_token_id)
    if eos_appended:
        completion_ids.append(int(eos_token_id))

    original_prompt_length = len(effective_prompt)
    original_completion_length = len(completion_ids)
    if len(effective_prompt) > max_prompt_length:
        effective_prompt = effective_prompt[-max_prompt_length:]
    if len(completion_ids) > max_completion_length:
        completion_ids = completion_ids[:max_completion_length]
    if len(effective_prompt) + len(completion_ids) > max_length:
        raise ValueError(
            "Prompt and completion still exceed max_length after applying individual limits."
        )
    if not effective_prompt or not completion_ids:
        raise ValueError("Prompt and completion must each contain at least one token.")

    return effective_prompt + completion_ids, len(effective_prompt), {
        "prompt_tokens": len(effective_prompt),
        "completion_tokens": len(completion_ids),
        "prompt_truncated": original_prompt_length > len(effective_prompt),
        "completion_truncated": original_completion_length > len(completion_ids),
        "eos_appended": eos_appended,
    }


class CompletionScorer:
    def __init__(self, model: Any, torch_module: Any, use_logits_to_keep: bool) -> None:
        self.model = model
        self.torch = torch_module
        self.use_logits_to_keep = use_logits_to_keep
        self.fallback_reason: str | None = None

    def score(self, token_ids: list[int], prompt_length: int) -> tuple[float, float]:
        torch = self.torch
        input_ids = torch.tensor([token_ids], dtype=torch.long, device="cuda:0")
        attention_mask = torch.ones_like(input_ids)
        completion_length = len(token_ids) - prompt_length
        kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": False,
        }
        if self.use_logits_to_keep:
            kwargs["logits_to_keep"] = completion_length + 1

        with torch.inference_mode():
            try:
                output = self.model(**kwargs)
            except (TypeError, ValueError) as exc:
                if not self.use_logits_to_keep or "logits_to_keep" not in str(exc):
                    raise
                self.use_logits_to_keep = False
                self.fallback_reason = str(exc)
                kwargs.pop("logits_to_keep")
                output = self.model(**kwargs)

            logits = output.logits
            if logits.shape[1] == completion_length + 1:
                completion_logits = logits[:, :-1, :]
            elif logits.shape[1] == len(token_ids):
                completion_logits = logits[:, prompt_length - 1 : -1, :]
            else:
                raise RuntimeError(
                    f"Unexpected logits length {logits.shape[1]} for input={len(token_ids)}, "
                    f"completion={completion_length}."
                )
            targets = input_ids[:, prompt_length:]
            token_log_probs = torch.log_softmax(completion_logits.float(), dim=-1)
            token_log_probs = token_log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            log_prob_sum = float(token_log_probs.sum().item())
            log_prob_mean = float(token_log_probs.mean().item())

        del input_ids, attention_mask, output, logits, completion_logits, targets, token_log_probs
        return log_prob_sum, log_prob_mean


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def report_table(groups: dict[str, dict[str, Any]], threshold: float) -> list[str]:
    lines = [
        "| 그룹 | N | mean: chosen/경계/rejected | sum: chosen/rejected | mean 평균 | mean 중앙값 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, summary in groups.items():
        counts = summary["preference_counts"]
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    str(summary["count"]),
                    f"{counts['chosen_above_band']}/{counts['near_zero_band']}/{counts['rejected_above_band']}",
                    f"{summary['sum_margin_sign_counts']['chosen_positive']}/{summary['sum_margin_sign_counts']['rejected_positive']}",
                    fmt(summary["mean_margin"]["mean"]),
                    fmt(summary["mean_margin"]["p50"]),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append(f"> 경계는 |mean margin| ≤ {threshold:g} nats/token인 진단용 구간입니다.")
    return lines


def write_report(path: Path, summary: dict[str, Any]) -> None:
    overall = summary["overall"]
    counts = overall["preference_counts"]
    sum_counts = overall["sum_margin_sign_counts"]
    threshold = float(summary["near_zero_threshold"])
    lines = [
        "# V2 step-SFT preference margin 분석",
        "",
        f"- 모델: `{summary['model']}`",
        f"- 데이터: `{summary['data_path']}`",
        f"- pair 수: {overall['count']}",
        f"- mean margin 평균/중앙값: {fmt(overall['mean_margin']['mean'])} / {fmt(overall['mean_margin']['p50'])}",
        f"- chosen 우세 / 경계 / rejected 우세: {counts['chosen_above_band']} / {counts['near_zero_band']} / {counts['rejected_above_band']}",
        f"- sum margin chosen/rejected 우세: {sum_counts['chosen_positive']} / {sum_counts['rejected_positive']}",
        f"- 길이 차이와 sum margin 상관: {fmt(overall['correlation_length_delta_sum_margin'])}",
        f"- 길이 차이와 mean margin 상관: {fmt(overall['correlation_length_delta_mean_margin'])}",
        "",
        "DPO가 사용하는 기본 pair log-ratio는 합산 sequence log-prob에 기반합니다. 다만 reference policy가 "
        "동일한 시작 SFT policy이므로 학습 시작점에서는 policy와 reference의 log-ratio 차이가 0으로 상쇄됩니다.",
        "mean margin은 `chosen 평균 token log-prob - rejected 평균 token log-prob`입니다. "
        "양수면 시작 SFT 모델이 chosen을 더 자연스럽게 보고, 음수면 rejected를 더 자연스럽게 봅니다.",
        "sum margin은 답변 길이 영향을 크게 받으므로 단독 필터 기준으로 사용하지 않습니다.",
        "",
        "## 카테고리",
        "",
        *report_table(summary["by_category"], threshold),
        "",
        "## 검수 결정",
        "",
        *report_table(summary["by_review_decision"], threshold),
        "",
        "## chosen 출처",
        "",
        *report_table(summary["by_chosen_source"], threshold),
        "",
        "## DPO split",
        "",
        *report_table(summary["by_split"], threshold),
        "",
        "## 주의",
        "",
        "이 값은 데이터 품질의 정답 판정이 아니라 시작 policy 관점의 난이도 진단입니다. "
        "음수 pair도 유용할 수 있고, 사람/상위 judge가 정한 정답 관계를 이 점수만으로 뒤집으면 안 됩니다.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    training = config["training"]
    gpu_device = args.gpu_device
    if gpu_device is None:
        gpu_device = str(training.get("cuda_visible_devices", "0"))
    if "," in gpu_device:
        raise ValueError("Margin analysis uses exactly one GPU; pass one --gpu-device.")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_device
    os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    import torch
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    from train.util import ensure_embedding_accessors, load_causal_lm_model

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this analysis.")
    set_seed(args.seed)

    raw_rows = read_preference_jsonl(config["data_path"])
    normalized_rows, dataset_stats = validate_preference_rows(raw_rows)
    raw_by_id = {row["id"]: row for row in raw_rows}
    train_rows, eval_rows, split_stats = split_preference_rows(
        normalized_rows,
        train_ratio=float(config["split"]["train_ratio"]),
        seed=args.seed,
    )
    split_by_id = {row["id"]: "train" for row in train_rows}
    split_by_id.update({row["id"]: "eval" for row in eval_rows})
    rows = normalized_rows[: args.limit] if args.limit else normalized_rows

    adapter_path = str(Path(args.model or config["model"]).resolve())
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    if not tokenizer.chat_template:
        raise ValueError(f"Adapter tokenizer has no chat template: {adapter_path}")
    adapter_config = PeftConfig.from_pretrained(adapter_path)
    model_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    base_model = load_causal_lm_model(
        AutoModelForCausalLM,
        adapter_config.base_model_name_or_path,
        model_dtype,
    )
    base_model.config.use_cache = False
    ensure_embedding_accessors(base_model)
    model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=False)
    model.eval()
    model.to("cuda:0")
    scorer = CompletionScorer(model, torch, args.use_logits_to_keep)

    max_prompt_length = int(training["max_prompt_length"])
    max_completion_length = int(training["max_completion_length"])
    max_length = int(training["max_length"])
    output_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        chosen_ids, chosen_prompt_length, chosen_meta = render_token_ids(
            tokenizer,
            row["prompt"],
            row["chosen"],
            max_prompt_length,
            max_completion_length,
            max_length,
        )
        rejected_ids, rejected_prompt_length, rejected_meta = render_token_ids(
            tokenizer,
            row["prompt"],
            row["rejected"],
            max_prompt_length,
            max_completion_length,
            max_length,
        )
        if chosen_ids[:chosen_prompt_length] != rejected_ids[:rejected_prompt_length]:
            raise RuntimeError(f"Chosen/rejected rendered prompts differ for id {row['id']!r}.")
        chosen_sum, chosen_mean = scorer.score(chosen_ids, chosen_prompt_length)
        rejected_sum, rejected_mean = scorer.score(rejected_ids, rejected_prompt_length)
        review = raw_by_id[row["id"]].get("review", {})
        mean_margin = chosen_mean - rejected_mean
        if mean_margin > args.near_zero_threshold:
            band = "chosen_above_band"
        elif mean_margin < -args.near_zero_threshold:
            band = "rejected_above_band"
        else:
            band = "near_zero_band"
        output_rows.append(
            {
                "id": row["id"],
                "category": row["category"],
                "split": split_by_id[row["id"]],
                "review_decision": review.get("decision", "unknown"),
                "chosen_source": review.get("chosen_source", "unknown"),
                "rejected_source": review.get("rejected_source", "unknown"),
                "chosen_sum_log_prob": chosen_sum,
                "rejected_sum_log_prob": rejected_sum,
                "sum_margin": chosen_sum - rejected_sum,
                "chosen_mean_log_prob": chosen_mean,
                "rejected_mean_log_prob": rejected_mean,
                "mean_margin": mean_margin,
                "margin_band": band,
                "prompt_tokens": chosen_meta["prompt_tokens"],
                "chosen_tokens": chosen_meta["completion_tokens"],
                "rejected_tokens": rejected_meta["completion_tokens"],
                "chosen_truncated": chosen_meta["completion_truncated"],
                "rejected_truncated": rejected_meta["completion_truncated"],
                "prompt_truncated": chosen_meta["prompt_truncated"],
            }
        )
        if index % args.progress_every == 0 or index == len(rows):
            print(
                json.dumps(
                    {
                        "completed": index,
                        "total": len(rows),
                        "last_id": row["id"],
                        "last_mean_margin": round(mean_margin, 6),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    summary = {
        "model": adapter_path,
        "base_model": adapter_config.base_model_name_or_path,
        "data_path": str(Path(config["data_path"]).resolve()),
        "gpu_device": gpu_device,
        "dtype": str(model_dtype),
        "near_zero_threshold": args.near_zero_threshold,
        "use_logits_to_keep": scorer.use_logits_to_keep,
        "logits_to_keep_fallback_reason": scorer.fallback_reason,
        "dataset_stats": dataset_stats,
        "split_stats": split_stats,
        "processed_rows": len(output_rows),
        "overall": summarize_rows(output_rows, args.near_zero_threshold),
        "by_category": grouped_summary(output_rows, "category", args.near_zero_threshold),
        "by_review_decision": grouped_summary(
            output_rows, "review_decision", args.near_zero_threshold
        ),
        "by_chosen_source": grouped_summary(
            output_rows, "chosen_source", args.near_zero_threshold
        ),
        "by_split": grouped_summary(output_rows, "split", args.near_zero_threshold),
    }
    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "margins.jsonl", output_rows)
    write_jsonl(
        output_dir / "review_priority.jsonl",
        sorted(output_rows, key=lambda row: (abs(float(row["mean_margin"])), str(row["id"]))),
    )
    write_json(output_dir / "summary.json", summary)
    write_report(output_dir / "report.md", summary)
    print(json.dumps({"output_dir": str(output_dir), "overall": summary["overall"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
