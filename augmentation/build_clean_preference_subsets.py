"""Build reproducible balanced DPO pilot subsets from reviewed preferences.

The script produces equal-size category-balanced control, high-confidence,
and high-confidence length-prioritized arms. The last arm is a best-available
diagnostic and does not claim strict length matching.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from augmentation.analyze_preference_margins import render_token_ids  # noqa: E402
from train.dpo_dataloader import (  # noqa: E402
    SUPPORTED_CATEGORIES,
    read_preference_jsonl,
    validate_preference_rows,
)


THRESHOLDS = (1.2, 1.33, 1.5, 2.0, 2.5, 3.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", default="google/gemma-4-E4B-it")
    parser.add_argument("--per-category", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--max-completion-length", type=int, default=1536)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def stable_seed(seed: int, arm: str, category: str) -> int:
    payload = f"{seed}:{arm}:{category}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def record_lengths(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    max_prompt_length: int,
    max_completion_length: int,
    max_length: int,
) -> list[dict[str, Any]]:
    records = []
    for row in rows:
        lengths = {}
        for field in ("chosen", "rejected"):
            _, _, metadata = render_token_ids(
                tokenizer=tokenizer,
                prompt=row["prompt"],
                answer=row[field],
                max_prompt_length=max_prompt_length,
                max_completion_length=max_completion_length,
                max_length=max_length,
            )
            lengths[field] = int(metadata["completion_tokens"])
        shorter = min(lengths.values())
        longer = max(lengths.values())
        records.append(
            {
                "row": row,
                "chosen_tokens": lengths["chosen"],
                "rejected_tokens": lengths["rejected"],
                "length_ratio": longer / shorter,
                "rejected_longer": lengths["rejected"] > lengths["chosen"],
            }
        )
    return records


def by_category(records: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["row"]["category"]].append(record)
    return grouped


def sample_balanced(
    records: Sequence[dict[str, Any]],
    per_category: int,
    seed: int,
    arm: str,
) -> list[dict[str, Any]]:
    selected = []
    grouped = by_category(records)
    for category in SUPPORTED_CATEGORIES:
        candidates = list(grouped.get(category, []))
        if len(candidates) < per_category:
            raise ValueError(
                f"{arm}: category {category!r} has {len(candidates)} rows, "
                f"fewer than --per-category={per_category}."
            )
        random.Random(stable_seed(seed, arm, category)).shuffle(candidates)
        selected.extend(candidates[:per_category])
    return sorted(selected, key=lambda item: int(item["row"]["id"]))


def select_high_matched(
    control: Sequence[dict[str, Any]],
    high_records: Sequence[dict[str, Any]],
    per_category: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Replace only medium-confidence control rows with high-confidence rows.

    Category totals and aggregate accept/swap counts remain fixed. Keeping all
    high-confidence control rows minimizes sample-identity differences between
    the two main pilot arms.
    """
    retained = [
        record
        for record in control
        if record["row"].get("review", {}).get("confidence") == "high"
    ]
    retained_ids = {str(record["row"]["id"]) for record in retained}
    grouped_retained = by_category(retained)
    available: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in high_records:
        if str(record["row"]["id"]) in retained_ids:
            continue
        category = record["row"]["category"]
        decision = record["row"].get("review", {}).get("decision", "missing")
        available[category][decision].append(record)

    slots = {
        category: per_category - len(grouped_retained.get(category, []))
        for category in SUPPORTED_CATEGORIES
    }
    control_decisions = Counter(
        record["row"].get("review", {}).get("decision", "missing")
        for record in control
    )
    retained_decisions = Counter(
        record["row"].get("review", {}).get("decision", "missing")
        for record in retained
    )
    target_swaps = (
        control_decisions.get("swap_pair", 0)
        - retained_decisions.get("swap_pair", 0)
    )
    original_medium_swaps = {
        category: sum(
            record["row"].get("review", {}).get("decision") == "swap_pair"
            and record["row"].get("review", {}).get("confidence") != "high"
            for record in control
            if record["row"]["category"] == category
        )
        for category in SUPPORTED_CATEGORIES
    }

    swap_options = []
    for category in SUPPORTED_CATEGORIES:
        max_swaps = min(slots[category], len(available[category]["swap_pair"]))
        min_swaps = max(
            0,
            slots[category] - len(available[category]["accept_pair"]),
        )
        swap_options.append(range(min_swaps, max_swaps + 1))
    allocations = [
        values
        for values in itertools.product(*swap_options)
        if sum(values) == target_swaps
    ]
    if not allocations:
        raise ValueError(
            "Cannot preserve aggregate accept/swap counts while replacing "
            "medium-confidence rows with high-confidence rows."
        )
    swap_allocation = min(
        allocations,
        key=lambda values: (
            sum(
                abs(values[index] - original_medium_swaps[category])
                for index, category in enumerate(SUPPORTED_CATEGORIES)
            ),
            values,
        ),
    )

    selected = list(retained)
    for index, category in enumerate(SUPPORTED_CATEGORIES):
        swap_count = swap_allocation[index]
        accept_count = slots[category] - swap_count
        for decision, count in (
            ("swap_pair", swap_count),
            ("accept_pair", accept_count),
        ):
            candidates = list(available[category][decision])
            random.Random(
                stable_seed(seed, f"high-matched-{decision}", category)
            ).shuffle(candidates)
            if len(candidates) < count:
                raise ValueError(
                    f"high_matched: {category!r}/{decision} has "
                    f"{len(candidates)} candidates, needs {count}."
                )
            selected.extend(candidates[:count])
    return sorted(selected, key=lambda item: int(item["row"]["id"]))


def select_length_priority(
    records: Sequence[dict[str, Any]],
    per_category: int,
    seed: int,
) -> list[dict[str, Any]]:
    selected = []
    grouped = by_category(records)
    for category in SUPPORTED_CATEGORIES:
        candidates = list(grouped.get(category, []))
        if len(candidates) < per_category:
            raise ValueError(
                f"high_length_priority: category {category!r} has {len(candidates)} rows, "
                f"fewer than --per-category={per_category}."
            )
        rng = random.Random(stable_seed(seed, "length-priority", category))
        tie_breakers = {str(item["row"]["id"]): rng.random() for item in candidates}
        candidates.sort(
            key=lambda item: (
                item["length_ratio"],
                tie_breakers[str(item["row"]["id"])],
            )
        )
        selected.extend(candidates[:per_category])
    return sorted(selected, key=lambda item: int(item["row"]["id"]))


def summarize(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    ratios = [float(record["length_ratio"]) for record in records]
    chosen = [int(record["chosen_tokens"]) for record in records]
    rejected = [int(record["rejected_tokens"]) for record in records]
    confidence = Counter(
        record["row"].get("review", {}).get("confidence", "missing")
        for record in records
    )
    decisions = Counter(
        record["row"].get("review", {}).get("decision", "missing")
        for record in records
    )
    return {
        "rows": len(records),
        "category_counts": dict(
            sorted(Counter(record["row"]["category"] for record in records).items())
        ),
        "confidence_counts": dict(sorted(confidence.items())),
        "decision_counts": dict(sorted(decisions.items())),
        "chosen_tokens_mean": round(statistics.fmean(chosen), 2),
        "rejected_tokens_mean": round(statistics.fmean(rejected), 2),
        "length_ratio_mean": round(statistics.fmean(ratios), 3),
        "length_ratio_p50": round(percentile(ratios, 0.5), 3),
        "length_ratio_p90": round(percentile(ratios, 0.9), 3),
        "length_ratio_max": round(max(ratios), 3),
        "rejected_longer_count": sum(record["rejected_longer"] for record in records),
    }


def feasibility(
    high_records: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for threshold in THRESHOLDS:
        selected = [
            record for record in high_records if record["length_ratio"] <= threshold
        ]
        counts = Counter(record["row"]["category"] for record in selected)
        output.append(
            {
                "max_ratio": threshold,
                "rows": len(selected),
                "category_counts": {
                    category: counts.get(category, 0)
                    for category in SUPPORTED_CATEGORIES
                },
                "balanced_per_category": min(
                    counts.get(category, 0)
                    for category in SUPPORTED_CATEGORIES
                ),
            }
        )
    return output


def write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record["row"], ensure_ascii=False) + "\n")


def markdown_report(payload: dict[str, Any]) -> str:
    per_category = payload["settings"]["per_category"]
    lines = [
        "# LogicKor DPO clean subset 선택 보고서",
        "",
        "## 선택 원칙",
        "",
        f"- 세 arm은 동일하게 카테고리당 {per_category}개를 사용한다.",
        "- control_balanced: 전체 reviewed pair에서 층화 무작위 추출",
        "- high_balanced: control의 high pair는 유지하고 medium pair만 같은 카테고리의 high pair로 교체",
        "- 두 주 arm은 카테고리 수와 전체 accept/swap 수가 같고, 공통 표본을 최대한 유지한다.",
        "- high_length_priority_balanced: high-confidence 중 카테고리별 token 길이 비가 가까운 순서로 추출",
        "- 길이 우선 arm은 source pool 내 최선의 진단용 subset이며 strict length-matched 데이터가 아니다.",
        "",
        "## Source 및 subset 요약",
        "",
        "| Arm | N | High | Medium | Accept | Swap | Chosen tok | Rejected tok | Ratio p50 | Ratio p90 | Ratio max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in (
        "source",
        "control_balanced",
        "high_balanced",
        "high_length_priority_balanced",
    ):
        summary = payload["summaries"][name]
        lines.append(
            f"| {name} | {summary['rows']} | "
            f"{summary['confidence_counts'].get('high', 0)} | "
            f"{summary['confidence_counts'].get('medium', 0)} | "
            f"{summary['decision_counts'].get('accept_pair', 0)} | "
            f"{summary['decision_counts'].get('swap_pair', 0)} | "
            f"{summary['chosen_tokens_mean']:.2f} | "
            f"{summary['rejected_tokens_mean']:.2f} | "
            f"{summary['length_ratio_p50']:.3f} | "
            f"{summary['length_ratio_p90']:.3f} | "
            f"{summary['length_ratio_max']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Strict length balance feasibility",
            "",
            "| Max token ratio | High rows | Balanced rows/category | "
            + " | ".join(SUPPORTED_CATEGORIES)
            + " |",
            "|---:|---:|---:|" + "---:|" * len(SUPPORTED_CATEGORIES),
        ]
    )
    for row in payload["length_feasibility"]:
        counts = row["category_counts"]
        lines.append(
            f"| {row['max_ratio']:.2f} | {row['rows']} | "
            f"{row['balanced_per_category']} | "
            + " | ".join(str(counts[category]) for category in SUPPORTED_CATEGORIES)
            + " |"
        )
    lines.extend(
        [
            "",
            "## 판단",
            "",
            "- token ratio 1.2 이하 high pair만으로는 6개 카테고리 균형 실험을 만들 수 없다.",
            "- 첫 pilot은 high_balanced를 주 arm으로 사용한다.",
            "- control_balanced는 confidence 효과 비교용이다.",
            f"- 두 주 arm은 {payload['main_arm_comparison']['overlap_rows']}개를 공유하고 "
            f"{payload['main_arm_comparison']['replaced_rows']}개만 교체한다.",
            "- high_length_priority_balanced는 길이 confound의 방향을 보는 보조 arm이다.",
            "- strict length-matched 실험은 기존 380개 필터링만으로 불가능하다.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if args.per_category < 2:
        raise ValueError("--per-category must be at least 2.")
    raw_rows = read_preference_jsonl(args.input)
    validate_preference_rows(raw_rows)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    records = record_lengths(
        raw_rows,
        tokenizer,
        args.max_prompt_length,
        args.max_completion_length,
        args.max_length,
    )
    high_records = [
        record
        for record in records
        if record["row"].get("review", {}).get("confidence") == "high"
    ]
    control = sample_balanced(records, args.per_category, args.seed, "control")
    high_matched = select_high_matched(
        control, high_records, args.per_category, args.seed
    )
    arms = {
        "control_balanced": control,
        "high_balanced": high_matched,
        "high_length_priority_balanced": select_length_priority(
            high_records, args.per_category, args.seed
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, selected in arms.items():
        write_jsonl(args.output_dir / f"{name}.jsonl", selected)

    control_ids = {str(record["row"]["id"]) for record in control}
    high_ids = {str(record["row"]["id"]) for record in high_matched}
    payload = {
        "source": str(Path(args.input)),
        "settings": {
            "tokenizer": args.tokenizer,
            "per_category": args.per_category,
            "seed": args.seed,
            "max_prompt_length": args.max_prompt_length,
            "max_completion_length": args.max_completion_length,
            "max_length": args.max_length,
        },
        "summaries": {
            "source": summarize(records),
            **{name: summarize(selected) for name, selected in arms.items()},
        },
        "length_feasibility": feasibility(high_records),
        "main_arm_comparison": {
            "overlap_rows": len(control_ids & high_ids),
            "replaced_rows": len(control_ids - high_ids),
            "category_counts_equal": Counter(
                record["row"]["category"] for record in control
            )
            == Counter(record["row"]["category"] for record in high_matched),
            "decision_counts_equal": Counter(
                record["row"].get("review", {}).get("decision", "missing")
                for record in control
            )
            == Counter(
                record["row"].get("review", {}).get("decision", "missing")
                for record in high_matched
            ),
        },
        "outputs": {
            name: str(args.output_dir / f"{name}.jsonl")
            for name in arms
        },
    }
    with (args.output_dir / "selection_report.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    (args.output_dir / "selection_report.md").write_text(
        markdown_report(payload),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
