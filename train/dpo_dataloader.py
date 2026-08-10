"""Validation and splitting for LogicKor preference JSONL data."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Sequence


PROMPT_ROLES = ("user", "assistant", "user")
SUPPORTED_CATEGORIES = ("추론", "수학", "코딩", "이해", "글쓰기", "문법")


def read_preference_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    rows: list[dict[str, Any]] = []
    try:
        handle = source.open(encoding="utf-8-sig")
    except OSError as exc:
        raise ValueError(f"Cannot read preference dataset {source}: {exc}") from exc
    with handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {source}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{source}:{line_number}: each row must be an object.")
            rows.append(row)
    return rows


def _non_empty_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string.")
    return value.strip()


def validate_preference_rows(rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate the reviewed prompt/chosen/rejected contract and normalize whitespace."""
    if not rows:
        raise ValueError("Preference dataset is empty.")

    normalized_rows: list[dict[str, Any]] = []
    seen_ids: set[int | str] = set()
    categories: Counter[str] = Counter()
    for row_index, row in enumerate(rows, 1):
        label = f"Row {row_index}"
        source_id = row.get("id")
        if isinstance(source_id, bool) or not isinstance(source_id, (int, str)) or source_id == "":
            raise ValueError(f"{label}: id must be a non-empty string or integer.")
        if source_id in seen_ids:
            raise ValueError(f"{label}: duplicate id {source_id!r}.")
        seen_ids.add(source_id)

        category = _non_empty_text(row.get("category"), f"{label}.category")
        if category not in SUPPORTED_CATEGORIES:
            raise ValueError(
                f"{label}.category must be one of {SUPPORTED_CATEGORIES}, got {category!r}."
            )
        prompt = row.get("prompt")
        if not isinstance(prompt, list) or len(prompt) != len(PROMPT_ROLES):
            raise ValueError(f"{label}.prompt must contain exactly three messages.")
        normalized_prompt: list[dict[str, str]] = []
        for message_index, (message, expected_role) in enumerate(zip(prompt, PROMPT_ROLES, strict=True)):
            message_label = f"{label}.prompt[{message_index}]"
            if not isinstance(message, dict) or set(message) != {"role", "content"}:
                raise ValueError(f"{message_label} must contain only role and content.")
            if message.get("role") != expected_role:
                raise ValueError(f"{message_label}.role must be {expected_role!r}.")
            normalized_prompt.append(
                {
                    "role": expected_role,
                    "content": _non_empty_text(message.get("content"), f"{message_label}.content"),
                }
            )

        chosen = _non_empty_text(row.get("chosen"), f"{label}.chosen")
        rejected = _non_empty_text(row.get("rejected"), f"{label}.rejected")
        if chosen == rejected:
            raise ValueError(f"{label}: chosen and rejected must differ.")

        normalized_rows.append(
            {
                "id": source_id,
                "category": category,
                "prompt": normalized_prompt,
                "chosen": chosen,
                "rejected": rejected,
            }
        )
        categories[category] += 1

    return normalized_rows, {
        "row_count": len(normalized_rows),
        "category_counts": dict(sorted(categories.items())),
        "unique_id_count": len(seen_ids),
    }


def split_preference_rows(
    rows: Sequence[dict[str, Any]], train_ratio: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Deterministically split while keeping every category in train and eval."""
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("split.train_ratio must be between 0 and 1.")
    if len(rows) < 2:
        raise ValueError("Preference train/eval split requires at least two rows.")

    indices_by_category: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        indices_by_category.setdefault(row["category"], []).append(index)
    undersized = [category for category, indices in indices_by_category.items() if len(indices) < 2]
    if undersized:
        raise ValueError(
            "Each category needs at least two preference rows for category-covered train/eval splits. "
            f"Undersized categories: {sorted(undersized)}"
        )

    rng = random.Random(seed)
    eval_indices: set[int] = set()
    protected_train_indices: set[int] = set()
    remaining_indices: list[int] = []
    for category in sorted(indices_by_category):
        category_indices = list(indices_by_category[category])
        rng.shuffle(category_indices)
        eval_indices.add(category_indices[0])
        protected_train_indices.add(category_indices[1])
        remaining_indices.extend(category_indices[2:])

    requested_eval_count = len(rows) - int(len(rows) * train_ratio)
    target_eval_count = max(len(indices_by_category), requested_eval_count)
    target_eval_count = min(target_eval_count, len(rows) - len(protected_train_indices))
    rng.shuffle(remaining_indices)
    eval_indices.update(remaining_indices[: target_eval_count - len(eval_indices)])
    train_indices = set(range(len(rows))) - eval_indices
    train_rows = [dict(row) for index, row in enumerate(rows) if index in train_indices]
    eval_rows = [dict(row) for index, row in enumerate(rows) if index in eval_indices]

    train_ids = {row["id"] for row in train_rows}
    eval_ids = {row["id"] for row in eval_rows}
    if train_ids & eval_ids:
        raise RuntimeError("Preference split leakage detected.")
    train_category_counts = Counter(row["category"] for row in train_rows)
    eval_category_counts = Counter(row["category"] for row in eval_rows)
    return train_rows, eval_rows, {
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "train_ids": len(train_ids),
        "eval_ids": len(eval_ids),
        "train_category_counts": dict(sorted(train_category_counts.items())),
        "eval_category_counts": dict(sorted(eval_category_counts.items())),
    }


def to_trl_preference_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert string completions to TRL's fully conversational preference format."""
    converted: list[dict[str, Any]] = []
    for row in rows:
        converted_row = {
            "prompt": [dict(message) for message in row["prompt"]],
            "chosen": [{"role": "assistant", "content": row["chosen"]}],
            "rejected": [{"role": "assistant", "content": row["rejected"]}],
        }
        if converted_row["chosen"][0]["role"] != "assistant" or converted_row["rejected"][0]["role"] != "assistant":
            raise RuntimeError("TRL preference completions must be assistant messages.")
        converted.append(converted_row)
    return converted
