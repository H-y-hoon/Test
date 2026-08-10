"""Validate category Codex reviews and build LogicKor DPO preference JSONL."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


SUPPORTED_CATEGORIES = ("추론", "수학", "코딩", "이해", "글쓰기", "문법")
DECISIONS = ("accept_pair", "swap_pair", "skip")
CONFIDENCE = ("high", "medium", "low")


class InputError(ValueError):
    """Raised when source candidates or review rows violate the contract."""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        handle = path.open(encoding="utf-8-sig")
    except OSError as exc:
        raise InputError(f"cannot read {path}: {exc}") from exc
    with handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise InputError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise InputError(f"{path}:{line_number}: each row must be an object")
            rows.append(row)
    return rows


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_id(value: Any, label: str) -> int | str:
    if isinstance(value, bool) or not isinstance(value, (int, str)) or value == "":
        raise InputError(f"{label}: invalid id {value!r}")
    return value


def validate_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InputError(f"{label}: expected non-empty text")
    return value


def candidate_id(candidate: dict[str, Any], label: str) -> str:
    worker_id = validate_text(candidate.get("worker_id"), f"{label}.worker_id")
    index = candidate.get("candidate_index")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise InputError(f"{label}.candidate_index must be a non-negative integer")
    return f"{worker_id}:{index}"


def load_source(path: Path) -> tuple[list[dict[str, Any]], dict[int | str, dict[str, Any]]]:
    rows = read_jsonl(path)
    if not rows:
        raise InputError(f"source is empty: {path}")
    source_map: dict[int | str, dict[str, Any]] = {}
    normalized_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        label = f"source row {index}"
        source_id = validate_id(row.get("id"), label)
        if source_id in source_map:
            raise InputError(f"{label}: duplicate id {source_id!r}")
        category = row.get("category")
        if category not in SUPPORTED_CATEGORIES:
            raise InputError(f"{label}: unsupported category {category!r}")
        prompt = row.get("prompt")
        if not isinstance(prompt, list) or len(prompt) != 3:
            raise InputError(f"{label}: prompt must contain exactly three messages")
        expected_roles = ("user", "assistant", "user")
        normalized_prompt: list[dict[str, str]] = []
        for message_index, (message, expected_role) in enumerate(zip(prompt, expected_roles, strict=True)):
            if not isinstance(message, dict) or message.get("role") != expected_role:
                raise InputError(f"{label}.prompt[{message_index}] must have role {expected_role}")
            normalized_prompt.append(
                {
                    "role": expected_role,
                    "content": validate_text(message.get("content"), f"{label}.prompt[{message_index}].content"),
                }
            )
        chosen = validate_text(row.get("chosen"), f"{label}.chosen")
        candidates = row.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise InputError(f"{label}.candidates must be a non-empty list")
        candidate_map: dict[str, dict[str, Any]] = {}
        for candidate_index_value, candidate in enumerate(candidates):
            if not isinstance(candidate, dict):
                raise InputError(f"{label}.candidates[{candidate_index_value}] must be an object")
            current_id = candidate_id(candidate, f"{label}.candidates[{candidate_index_value}]")
            if current_id in candidate_map:
                raise InputError(f"{label}: duplicate candidate id {current_id!r}")
            validate_text(candidate.get("text"), f"{label}.candidates[{candidate_index_value}].text")
            candidate_map[current_id] = candidate
        normalized = {
            "id": source_id,
            "category": category,
            "prompt": normalized_prompt,
            "chosen": chosen,
            "candidates": candidate_map,
        }
        normalized_rows.append(normalized)
        source_map[source_id] = normalized
    return normalized_rows, source_map


def load_reviews(paths: list[Path], source_map: dict[int | str, dict[str, Any]]) -> dict[int | str, dict[str, Any]]:
    reviews: dict[int | str, dict[str, Any]] = {}
    for path in paths:
        for index, row in enumerate(read_jsonl(path), 1):
            label = f"{path} row {index}"
            source_id = validate_id(row.get("id"), label)
            if source_id in reviews:
                raise InputError(f"{label}: duplicate review id {source_id!r}")
            if source_id not in source_map:
                raise InputError(f"{label}: id is absent from source")
            source = source_map[source_id]
            if row.get("category") != source["category"]:
                raise InputError(f"{label}: category does not match source")
            decision = row.get("decision")
            if decision not in DECISIONS:
                raise InputError(f"{label}: invalid decision {decision!r}")
            chosen_source = row.get("chosen_source")
            rejected_source = row.get("rejected_source")
            confidence = row.get("confidence")
            if confidence not in CONFIDENCE:
                raise InputError(f"{label}: invalid confidence {confidence!r}")
            validate_text(row.get("gemma_status"), f"{label}.gemma_status")
            validate_text(row.get("reason"), f"{label}.reason")
            if decision == "accept_pair":
                if chosen_source != "original" or rejected_source not in source["candidates"]:
                    raise InputError(f"{label}: accept_pair must use original vs a candidate")
            elif decision == "swap_pair":
                if chosen_source not in source["candidates"] or rejected_source != "original":
                    raise InputError(f"{label}: swap_pair must use a candidate vs original")
            elif chosen_source != "original" or rejected_source is not None:
                raise InputError(f"{label}: skip must use chosen_source=original and rejected_source=null")
            reviews[source_id] = row
    return reviews


def resolve_text(source: dict[str, Any], source_name: str, label: str) -> str:
    if source_name == "original":
        return source["chosen"]
    candidate = source["candidates"].get(source_name)
    if candidate is None:
        raise InputError(f"{label}: candidate {source_name!r} does not exist")
    if candidate.get("finish_reason") == "length":
        raise InputError(f"{label}: candidate {source_name!r} reached the generation length limit")
    return candidate["text"]


def finalize(
    source_rows: list[dict[str, Any]], reviews: dict[int | str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_ids = {row["id"] for row in source_rows}
    missing = source_ids - set(reviews)
    if missing:
        raise InputError(f"reviews do not cover all source ids: {sorted(missing, key=str)[:20]}")
    output_rows: list[dict[str, Any]] = []
    decision_counts: Counter[str] = Counter()
    confidence_counts: Counter[str] = Counter()
    category_pairs: Counter[str] = Counter()
    for source in source_rows:
        review = reviews[source["id"]]
        decision_counts[review["decision"]] += 1
        confidence_counts[review["confidence"]] += 1
        if review["decision"] == "skip":
            continue
        label = f"review id {source['id']}"
        chosen = resolve_text(source, review["chosen_source"], label)
        rejected = resolve_text(source, review["rejected_source"], label)
        if chosen.strip() == rejected.strip():
            raise InputError(f"{label}: resolved chosen and rejected are identical")
        output_rows.append(
            {
                "id": source["id"],
                "category": source["category"],
                "prompt": source["prompt"],
                "chosen": chosen,
                "rejected": rejected,
                "review": {
                    "decision": review["decision"],
                    "chosen_source": review["chosen_source"],
                    "rejected_source": review["rejected_source"],
                    "gemma_status": review["gemma_status"],
                    "reason": review["reason"],
                    "confidence": review["confidence"],
                },
            }
        )
        category_pairs[source["category"]] += 1
    return output_rows, {
        "source_rows": len(source_rows),
        "review_rows": len(reviews),
        "preference_pairs": len(output_rows),
        "decision_counts": dict(decision_counts),
        "confidence_counts": dict(confidence_counts),
        "category_pair_counts": dict(category_pairs),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--review", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.output.resolve() == args.report.resolve():
            raise InputError("--output and --report must be different files")
        source_rows, source_map = load_source(args.source)
        reviews = load_reviews(args.review, source_map)
        output_rows, report = finalize(source_rows, reviews)
        report.update(
            {
                "source": str(args.source),
                "reviews": [str(path) for path in args.review],
                "output": str(args.output),
            }
        )
        write_jsonl_atomic(args.output, output_rows)
        write_json_atomic(args.report, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except (InputError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
