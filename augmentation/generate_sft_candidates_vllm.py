"""Prepare, generate, and merge on-policy SFT response candidates with vLLM.

The generator deliberately sends only the existing conversation prefix
(`user -> assistant -> user`) to the model.  The reference/chosen answer is
kept in the task record for later judging, but is never included in the model
input.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence


SUPPORTED_CATEGORIES = ("추론", "수학", "코딩", "이해", "글쓰기", "문법")


class InputError(ValueError):
    """Raised when a task or result file violates the pipeline contract."""


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


def validate_id(source_id: Any, label: str) -> int | str:
    if not isinstance(source_id, (int, str)) or isinstance(source_id, bool) or source_id == "":
        raise InputError(f"{label}: invalid id {source_id!r}")
    return source_id


def validate_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InputError(f"{label}: expected non-empty text")
    return value


def normalize_task(row: dict[str, Any], index: int) -> dict[str, Any]:
    """Accept the source v2 schema or the prepared prompt/chosen schema."""
    label = f"task row {index}"
    source_id = validate_id(row.get("id"), label)
    category = row.get("category")
    if category not in SUPPORTED_CATEGORIES:
        raise InputError(f"{label} id {source_id}: unsupported category {category!r}")

    if "prompt" in row or "chosen" in row:
        prompt = row.get("prompt")
        chosen = validate_text(row.get("chosen"), f"{label} id {source_id} chosen")
        if not isinstance(prompt, list) or len(prompt) != 3:
            raise InputError(f"{label} id {source_id}: prompt must contain exactly three messages")
        normalized_prompt: list[dict[str, str]] = []
        expected_roles = ("user", "assistant", "user")
        for message_index, (message, expected_role) in enumerate(zip(prompt, expected_roles, strict=True)):
            if not isinstance(message, dict) or message.get("role") != expected_role:
                raise InputError(
                    f"{label} id {source_id}: prompt message {message_index} must have role {expected_role}"
                )
            normalized_prompt.append(
                {
                    "role": expected_role,
                    "content": validate_text(
                        message.get("content"),
                        f"{label} id {source_id} prompt message {message_index}",
                    ),
                }
            )
    else:
        questions = row.get("questions")
        references = row.get("references")
        if not isinstance(questions, list) or len(questions) != 2:
            raise InputError(f"{label} id {source_id}: questions must contain exactly two strings")
        if not isinstance(references, list) or len(references) != 2:
            raise InputError(f"{label} id {source_id}: references must contain exactly two strings")
        question_1 = validate_text(questions[0], f"{label} id {source_id} question 1")
        question_2 = validate_text(questions[1], f"{label} id {source_id} question 2")
        answer_1 = validate_text(references[0], f"{label} id {source_id} reference 1")
        chosen = validate_text(references[1], f"{label} id {source_id} reference 2")
        normalized_prompt = [
            {"role": "user", "content": question_1},
            {"role": "assistant", "content": answer_1},
            {"role": "user", "content": question_2},
        ]

    return {
        "id": source_id,
        "category": category,
        "prompt": normalized_prompt,
        "chosen": chosen,
    }


def load_tasks(path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if not rows:
        raise InputError(f"task file is empty: {path}")
    tasks = [normalize_task(row, index) for index, row in enumerate(rows, 1)]
    counts = Counter(task["id"] for task in tasks)
    duplicates = [source_id for source_id, count in counts.items() if count > 1]
    if duplicates:
        raise InputError(f"duplicate task ids: {sorted(duplicates, key=str)[:10]}")
    return tasks


def rows_by_id(rows: Sequence[dict[str, Any]], label: str) -> dict[int | str, dict[str, Any]]:
    result: dict[int | str, dict[str, Any]] = {}
    for index, row in enumerate(rows, 1):
        source_id = validate_id(row.get("id"), f"{label} row {index}")
        if source_id in result:
            raise InputError(f"duplicate {label} id {source_id!r}")
        result[source_id] = row
    return result


def normalized_candidate_text(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def parse_id_file(path: Path) -> set[int | str]:
    rows = read_jsonl(path)
    return {validate_id(row.get("id"), f"retry row {index}") for index, row in enumerate(rows, 1)}


def validate_candidate_row(
    row: dict[str, Any],
    *,
    expected_count: int | None = None,
    expected_worker: str | None = None,
) -> None:
    source_id = validate_id(row.get("id"), "candidate row")
    candidates = row.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise InputError(f"candidate id {source_id}: candidates must be a non-empty list")
    if expected_count is not None and len(candidates) != expected_count:
        raise InputError(
            f"candidate id {source_id}: expected {expected_count} candidates, got {len(candidates)}"
        )
    normalized: set[str] = set()
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise InputError(f"candidate id {source_id}: candidate {index} must be an object")
        text = validate_text(candidate.get("text"), f"candidate id {source_id} item {index}")
        if expected_worker is not None and candidate.get("worker_id") != expected_worker:
            raise InputError(
                f"candidate id {source_id}: item {index} has unexpected worker_id "
                f"{candidate.get('worker_id')!r}"
            )
        key = normalized_candidate_text(text)
        if key in normalized:
            raise InputError(f"candidate id {source_id}: duplicate candidate text within worker output")
        normalized.add(key)


def candidate_row_is_complete(row: dict[str, Any], expected_count: int, worker_id: str) -> bool:
    try:
        validate_candidate_row(row, expected_count=expected_count, expected_worker=worker_id)
    except InputError:
        return False
    return True


def chunks(rows: Sequence[dict[str, Any]], size: int) -> Iterable[Sequence[dict[str, Any]]]:
    for index in range(0, len(rows), size):
        yield rows[index : index + size]


def prepare_command(args: argparse.Namespace) -> int:
    tasks = load_tasks(args.source)
    selected = tasks
    if args.pilot_per_category is not None:
        if args.pilot_per_category < 1:
            raise InputError("--pilot-per-category must be positive")
        rng = random.Random(args.selection_seed)
        selected = []
        for category in SUPPORTED_CATEGORIES:
            category_tasks = [task for task in tasks if task["category"] == category]
            if len(category_tasks) < args.pilot_per_category:
                raise InputError(
                    f"category {category} has {len(category_tasks)} tasks; "
                    f"cannot select {args.pilot_per_category}"
                )
            selected.extend(rng.sample(category_tasks, args.pilot_per_category))
        original_order = {task["id"]: index for index, task in enumerate(tasks)}
        selected.sort(key=lambda task: original_order[task["id"]])

    if args.output.exists() and not args.overwrite:
        raise InputError(f"output already exists: {args.output}; pass --overwrite to replace it")
    write_jsonl_atomic(args.output, selected)
    print(
        json.dumps(
            {
                "source": str(args.source),
                "output": str(args.output),
                "tasks": len(selected),
                "categories": dict(Counter(task["category"] for task in selected)),
                "selection_seed": args.selection_seed if args.pilot_per_category is not None else None,
            },
            ensure_ascii=False,
        )
    )
    return 0


def build_llm_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": args.model,
        "dtype": args.dtype,
        "max_model_len": args.max_model_len,
        "tensor_parallel_size": 1,
        "seed": args.seed,
        "enforce_eager": args.enforce_eager,
        "trust_remote_code": args.trust_remote_code,
        "model_impl": args.model_impl,
    }
    if args.gpu_memory_utilization is not None:
        kwargs["gpu_memory_utilization"] = args.gpu_memory_utilization
    if args.cpu_offload_gb is not None:
        kwargs["cpu_offload_gb"] = args.cpu_offload_gb
    if args.language_model_only:
        kwargs["language_model_only"] = True
    if args.lora_path is not None:
        kwargs["enable_lora"] = True
        kwargs["max_lora_rank"] = args.max_lora_rank
    if args.disable_weight_tracking:
        kwargs["model_loader_extra_config"] = {"enable_weights_track": False}
    return kwargs


def build_sampling_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "n": args.candidate_count,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
    }
    if args.top_k is not None:
        kwargs["top_k"] = args.top_k
    if args.min_p is not None:
        kwargs["min_p"] = args.min_p
    if args.stop:
        kwargs["stop"] = args.stop
    return kwargs


def parse_vllm_output(
    task: dict[str, Any],
    output: Any,
    *,
    worker_id: str,
    seed: int,
    expected_count: int,
) -> dict[str, Any]:
    completions = getattr(output, "outputs", None)
    if not isinstance(completions, list) or len(completions) != expected_count:
        actual = len(completions) if isinstance(completions, list) else 0
        raise InputError(f"expected {expected_count} completions, got {actual}")

    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate_index, completion in enumerate(completions):
        text = validate_text(getattr(completion, "text", None), f"completion {candidate_index}").strip()
        key = normalized_candidate_text(text)
        if key in seen:
            raise InputError("worker returned duplicate candidate texts")
        seen.add(key)
        token_ids = getattr(completion, "token_ids", None)
        candidates.append(
            {
                "candidate_index": candidate_index,
                "worker_id": worker_id,
                "seed": seed,
                "text": text,
                "finish_reason": getattr(completion, "finish_reason", None),
                "stop_reason": getattr(completion, "stop_reason", None),
                "output_tokens": len(token_ids) if token_ids is not None else None,
            }
        )
    return {"id": task["id"], "category": task["category"], "candidates": candidates}


def generate_batch(
    llm: Any,
    sampling_params: Any,
    batch: Sequence[dict[str, Any]],
    *,
    worker_id: str,
    seed: int,
    candidate_count: int,
    lora_request: Any | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    conversations = [task["prompt"] for task in batch]
    try:
        outputs = llm.chat(
            conversations,
            sampling_params=sampling_params,
            use_tqdm=False,
            lora_request=lora_request,
        )
        if len(outputs) != len(batch):
            raise RuntimeError(f"vLLM returned {len(outputs)} outputs for {len(batch)} tasks")
    except Exception as batch_exc:
        generated: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        for task in batch:
            try:
                single_outputs = llm.chat(
                    [task["prompt"]],
                    sampling_params=sampling_params,
                    use_tqdm=False,
                    lora_request=lora_request,
                )
                if len(single_outputs) != 1:
                    raise RuntimeError(f"vLLM returned {len(single_outputs)} outputs for one task")
                generated.append(
                    parse_vllm_output(
                        task,
                        single_outputs[0],
                        worker_id=worker_id,
                        seed=seed,
                        expected_count=candidate_count,
                    )
                )
            except Exception as exc:
                failures.append(
                    {
                        "id": task["id"],
                        "category": task["category"],
                        "worker_id": worker_id,
                        "stage": "generation",
                        "reason": (
                            f"batch={type(batch_exc).__name__}: {batch_exc}; "
                            f"single={type(exc).__name__}: {exc}"
                        )[:1000],
                    }
                )
        return generated, failures

    generated = []
    failures = []
    for task, output in zip(batch, outputs, strict=True):
        try:
            generated.append(
                parse_vllm_output(
                    task,
                    output,
                    worker_id=worker_id,
                    seed=seed,
                    expected_count=candidate_count,
                )
            )
        except (InputError, AttributeError, TypeError) as exc:
            failures.append(
                {
                    "id": task["id"],
                    "category": task["category"],
                    "worker_id": worker_id,
                    "stage": "generation",
                    "reason": f"{type(exc).__name__}: {exc}"[:1000],
                }
            )
    return generated, failures


def generate_command(args: argparse.Namespace) -> int:
    if args.candidate_count < 1 or args.batch_size < 1:
        raise InputError("--candidate-count and --batch-size must be positive")
    if args.max_model_len < 1 or args.max_tokens < 1:
        raise InputError("--max-model-len and --max-tokens must be positive")
    if args.max_tokens >= args.max_model_len:
        raise InputError("--max-tokens must be smaller than --max-model-len")
    if not 0 <= args.temperature:
        raise InputError("--temperature must be non-negative")
    if not 0 < args.top_p <= 1:
        raise InputError("--top-p must be in (0, 1]")
    if args.gpu_memory_utilization is not None and not 0 < args.gpu_memory_utilization <= 1:
        raise InputError("--gpu-memory-utilization must be in (0, 1]")
    if args.lora_path is not None and not args.lora_path.is_dir():
        raise InputError(f"--lora-path is not a directory: {args.lora_path}")
    if args.max_lora_rank < 1:
        raise InputError("--max-lora-rank must be positive")
    if not args.gpu_device.isdigit():
        raise InputError("--gpu-device must be one numeric physical GPU index")
    if args.output.resolve() == args.failures.resolve():
        raise InputError("--output and --failures must be different files")

    tasks = load_tasks(args.tasks)
    task_by_id = {task["id"]: task for task in tasks}
    existing_rows = read_jsonl(args.output) if args.output.exists() else []
    existing = rows_by_id(existing_rows, "worker output")
    unknown = set(existing) - set(task_by_id)
    if unknown:
        raise InputError(f"worker output contains ids absent from tasks: {sorted(unknown, key=str)[:10]}")

    retry_ids: set[int | str] | None = None
    if args.retry_ids is not None:
        retry_ids = parse_id_file(args.retry_ids)
        unknown_retry = retry_ids - set(task_by_id)
        if unknown_retry:
            raise InputError(f"retry file contains ids absent from tasks: {sorted(unknown_retry, key=str)[:10]}")

    complete_ids = {
        source_id
        for source_id, row in existing.items()
        if candidate_row_is_complete(row, args.candidate_count, args.worker_id)
    }
    if retry_ids is None:
        pending = [task for task in tasks if task["id"] not in complete_ids]
    else:
        pending = [task for task in tasks if task["id"] in retry_ids]

    existing_failure_rows = read_jsonl(args.failures) if args.failures.exists() else []
    failure_map = rows_by_id(existing_failure_rows, "worker failure")
    generation_summary = {
        "tasks": len(tasks),
        "completed": len(complete_ids),
        "pending": len(pending),
        "retry_mode": retry_ids is not None,
        "worker_id": args.worker_id,
        "gpu_device": args.gpu_device,
        "model": args.model,
        "lora_path": str(args.lora_path) if args.lora_path is not None else None,
        "max_lora_rank": args.max_lora_rank if args.lora_path is not None else None,
        "candidate_count": args.candidate_count,
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "max_model_len": args.max_model_len,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "seed": args.seed,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "disable_weight_tracking": args.disable_weight_tracking,
    }
    if args.dry_run or not pending:
        print(json.dumps(generation_summary, ensure_ascii=False))
        return 0

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(**build_llm_kwargs(args))
    lora_request = (
        LoRARequest("sft_adapter", 1, str(args.lora_path), base_model_name=args.model)
        if args.lora_path is not None
        else None
    )
    sampling_params = SamplingParams(**build_sampling_kwargs(args))
    print(json.dumps(generation_summary, ensure_ascii=False), flush=True)

    for batch in chunks(pending, args.batch_size):
        generated, failures = generate_batch(
            llm,
            sampling_params,
            batch,
            worker_id=args.worker_id,
            seed=args.seed,
            candidate_count=args.candidate_count,
            lora_request=lora_request,
        )
        for row in generated:
            existing[row["id"]] = row
            failure_map.pop(row["id"], None)
        for row in failures:
            failure_map[row["id"]] = row

        ordered_output = [existing[task["id"]] for task in tasks if task["id"] in existing]
        ordered_failures = [failure_map[task["id"]] for task in tasks if task["id"] in failure_map]
        write_jsonl_atomic(args.output, ordered_output)
        write_jsonl_atomic(args.failures, ordered_failures)
        print(
            json.dumps(
                {
                    "batch_ids": [task["id"] for task in batch],
                    "generated": len(generated),
                    "failed": len(failures),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return 0


def merge_command(args: argparse.Namespace) -> int:
    if args.required_candidates < 1:
        raise InputError("--required-candidates must be positive")
    if len(args.worker_output) < 1:
        raise InputError("at least one --worker-output is required")
    if args.output.resolve() == args.incomplete.resolve():
        raise InputError("--output and --incomplete must be different files")

    tasks = load_tasks(args.tasks)
    task_ids = {task["id"] for task in tasks}
    worker_maps: list[tuple[Path, dict[int | str, dict[str, Any]]]] = []
    for path in args.worker_output:
        worker_map = rows_by_id(read_jsonl(path), f"worker output {path}")
        unknown = set(worker_map) - task_ids
        if unknown:
            raise InputError(f"{path} contains ids absent from tasks: {sorted(unknown, key=str)[:10]}")
        for row in worker_map.values():
            validate_candidate_row(row)
        worker_maps.append((path, worker_map))

    merged: list[dict[str, Any]] = []
    incomplete: list[dict[str, Any]] = []
    for task in tasks:
        candidates: list[dict[str, Any]] = []
        normalized_seen: set[str] = set()
        duplicate_count = 0
        missing_workers: list[str] = []
        for path, worker_map in worker_maps:
            row = worker_map.get(task["id"])
            if row is None:
                missing_workers.append(str(path))
                continue
            for candidate in row["candidates"]:
                key = normalized_candidate_text(candidate["text"])
                if key in normalized_seen:
                    duplicate_count += 1
                    continue
                normalized_seen.add(key)
                candidates.append(candidate)

        if len(candidates) < args.required_candidates:
            incomplete.append(
                {
                    "id": task["id"],
                    "category": task["category"],
                    "unique_candidates": len(candidates),
                    "required_candidates": args.required_candidates,
                    "duplicate_candidates": duplicate_count,
                    "missing_worker_outputs": missing_workers,
                    "reason": "insufficient unique candidates",
                }
            )
            continue
        merged.append({**task, "candidates": candidates[: args.required_candidates]})

    write_jsonl_atomic(args.output, merged)
    write_jsonl_atomic(args.incomplete, incomplete)
    print(
        json.dumps(
            {
                "tasks": len(tasks),
                "merged": len(merged),
                "incomplete": len(incomplete),
                "required_candidates": args.required_candidates,
                "output": str(args.output),
                "incomplete_output": str(args.incomplete),
            },
            ensure_ascii=False,
        )
    )
    return 0


def add_generate_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("generate", help="generate candidates for one GPU worker")
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--failures", type=Path, required=True)
    parser.add_argument("--retry-ids", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--lora-path", type=Path)
    parser.add_argument("--max-lora-rank", type=int, default=32)
    parser.add_argument("--gpu-device", required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--candidate-count", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--min-p", type=float)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gpu-memory-utilization", type=float)
    parser.add_argument("--cpu-offload-gb", type=float)
    parser.add_argument("--model-impl", choices=("auto", "transformers"), default="auto")
    parser.add_argument("--stop", action="append", default=[])
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--language-model-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--disable-weight-tracking", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.set_defaults(func=generate_command)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="normalize source rows and optionally select a pilot")
    prepare.add_argument("--source", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--pilot-per-category", type=int)
    prepare.add_argument("--selection-seed", type=int, default=42)
    prepare.add_argument("--overwrite", action="store_true")
    prepare.set_defaults(func=prepare_command)

    add_generate_parser(subparsers)

    merge = subparsers.add_parser("merge", help="merge and deduplicate worker candidate files")
    merge.add_argument("--tasks", type=Path, required=True)
    merge.add_argument("--worker-output", type=Path, action="append", required=True)
    merge.add_argument("--required-candidates", type=int, required=True)
    merge.add_argument("--output", type=Path, required=True)
    merge.add_argument("--incomplete", type=Path, required=True)
    merge.set_defaults(func=merge_command)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return args.func(args)
    except (InputError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
