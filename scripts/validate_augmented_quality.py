#!/usr/bin/env python3
import argparse
import json
import re
import statistics
import string
import tempfile
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


EXPECTED_KEYS = {"id", "category", "questions", "references"}
MIN_Q_UNIQUE_RATIO = 0.55
MIN_REF_UNIQUE_RATIO = 0.45
MIN_REF_MEAN_RATIO = 0.60
MAX_REF_PREFIX_SHARE = 0.20
STRICT_META_PATTERNS = [
    ("writing meta lead-in", re.compile(r"(고친\s*문장은\s*다음과\s*같다|작성하면\s*다음과\s*같다|다음과\s*같이\s*쓸\s*수\s*있다|요청에\s*맞춰)")),
]
FENCED_CODE_PATTERN = re.compile(r"```.*?```", re.DOTALL)
INLINE_CODE_PATTERN = re.compile(r"`[^`\n]+`")
QUOTED_CASE_PARTICLE_TOPIC_PATTERN = re.compile(r"[‘“][^’”\n]{1,120}(?:을|를|이|가)[’”]은")

FORBIDDEN_PATTERNS = [
    ("previous-case marker", re.compile(r"이전\s*사례")),
    (
        "numeric condition marker",
        re.compile(r"(?:구별되는\s*조건|조건\s*확장)(?:\s*\d+)?|조건\s*\d+(?=\s|$|[.)번:])"),
    ),
    ("generated review id", re.compile(r"(후속\s*검토|반례\s*점검|문체\s*전환|조건\s*확장)\s*\d+")),
    ("unfilled placeholder", re.compile(r"\{[A-Za-z_가-힣][A-Za-z0-9_가-힣-]{0,24}\}")),
    (
        "known particle artifact",
        re.compile(
            r"(도입를|공공성와|운영\s*재원가|자율성와|평가\s*공정성가|규칙를|예문를|판별법를|살리기은|라이선스을)"
        ),
    ),
    ("malformed grammar pair particle", re.compile(r"(던/든는|안/않는|왠/웬는)")),
    (
        "generic code placeholder",
        re.compile(r"(실제\s+.*요구에\s+맞게\s+.*바꾸면\s+된다|문제\s+요구에\s+맞게\s+수정하면\s+된다)"),
    ),
]


def load_jsonl(path):
    rows = []
    errors = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{path}:{line_no}: invalid JSON: {exc}")
                continue
            if not isinstance(row, dict):
                errors.append(f"{path}:{line_no}: row is not an object")
                continue
            row["_line"] = line_no
            rows.append(row)
    return rows, errors


def normalize(text, strip_numbers=False):
    text = unicodedata.normalize("NFKC", text).lower()
    if strip_numbers:
        text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans({c: " " for c in string.punctuation}))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def text_values(row):
    for field in ("questions", "references"):
        value = row.get(field)
        if isinstance(value, list):
            for idx, text in enumerate(value):
                if isinstance(text, str):
                    yield field, idx, text


def forbidden_findings(text):
    for label, pattern in FORBIDDEN_PATTERNS:
        scan_text = text
        if label == "unfilled placeholder":
            scan_text = FENCED_CODE_PATTERN.sub("", scan_text)
            scan_text = INLINE_CODE_PATTERN.sub("", scan_text)
        matches = list(pattern.finditer(scan_text))
        if label == "unfilled placeholder":
            matches = [
                match
                for match in matches
                if not (
                    len(match.group(0)) == 3
                    and match.group(0)[1].isascii()
                    and match.start() > 0
                    and scan_text[match.start() - 1] == "="
                )
            ]
        if matches:
            yield label


def validate(reference_path, candidate_path, expected_per_category, strict_v3=False):
    reference_rows, errors = load_jsonl(reference_path)
    candidate_rows, candidate_errors = load_jsonl(candidate_path)
    errors.extend(candidate_errors)
    hard_errors = list(errors)
    warnings = []
    hard_examples = []

    reference_categories = sorted({row.get("category") for row in reference_rows})
    reference_questions = {
        question
        for row in reference_rows
        for question in row.get("questions", [])
        if isinstance(question, str)
    }

    candidate_counts = Counter()
    ids = Counter()
    exact_questions = Counter()
    by_category = defaultdict(list)

    for row in candidate_rows:
        line = row.get("_line", "?")
        public_keys = set(row) - {"_line"}
        if public_keys != EXPECTED_KEYS:
            hard_errors.append(f"line {line}: keys must be exactly {sorted(EXPECTED_KEYS)}, got {sorted(public_keys)}")

        category = row.get("category")
        candidate_counts[category] += 1
        by_category[category].append(row)
        ids[row.get("id")] += 1

        if category not in reference_categories:
            hard_errors.append(f"line {line}: unknown category {category!r}")

        questions = row.get("questions")
        references = row.get("references")
        if not isinstance(questions, list) or not isinstance(references, list):
            hard_errors.append(f"line {line}: questions/references must be lists")
            continue
        if len(questions) != 2 or len(references) != 2:
            hard_errors.append(f"line {line}: questions/references must both have length 2")
        for field, values in (("questions", questions), ("references", references)):
            for idx, value in enumerate(values):
                if not isinstance(value, str) or not value.strip():
                    hard_errors.append(f"line {line}: {field}[{idx}] must be a non-empty string")

        for question in questions:
            if isinstance(question, str):
                exact_questions[question] += 1
                if question in reference_questions:
                    hard_examples.append(f"line {line}: source question copied: {question[:120]}")

        for field, idx, text in text_values(row):
            for label in forbidden_findings(text):
                hard_examples.append(f"line {line}: {label} in {field}[{idx}]: {text[:120]}")

        if strict_v3:
            strict_hard, strict_warnings = strict_category_findings(row)
            hard_examples.extend(f"line {line}: {issue}" for issue in strict_hard)
            warnings.extend(f"line {line}: {issue}" for issue in strict_warnings)

    duplicate_ids = [item for item, count in ids.items() if count > 1]
    if duplicate_ids:
        hard_errors.append(f"duplicate ids: {duplicate_ids[:10]}")

    duplicate_questions = [text for text, count in exact_questions.items() if count > 1]
    if duplicate_questions:
        hard_errors.append(f"duplicate exact questions: {len(duplicate_questions)}")
        hard_examples.extend(f"duplicate question: {text[:120]}" for text in duplicate_questions[:5])

    if expected_per_category:
        for category in reference_categories:
            count = candidate_counts.get(category, 0)
            if count != expected_per_category:
                hard_errors.append(f"{category}: expected {expected_per_category} rows, got {count}")

    reference_metrics = category_metrics(group_by_category(reference_rows))
    candidate_metrics = category_metrics(by_category)
    for category, metrics in candidate_metrics.items():
        ref = reference_metrics.get(category)
        if not ref:
            continue
        if metrics["q_unique_ratio"] < MIN_Q_UNIQUE_RATIO:
            warnings.append(f"{category}: question template diversity looks low ({metrics['q_unique_ratio']:.2f})")
        if metrics["ref_unique_ratio"] < MIN_REF_UNIQUE_RATIO:
            warnings.append(f"{category}: reference template diversity looks low ({metrics['ref_unique_ratio']:.2f})")
        if metrics["ref_mean"] < ref["ref_mean"] * MIN_REF_MEAN_RATIO:
            warnings.append(
                f"{category}: reference mean may be too short ({metrics['ref_mean']:.1f} vs high {ref['ref_mean']:.1f})"
            )
        if metrics["ref_count"] >= 10 and metrics["top_ref_prefix_share"] > MAX_REF_PREFIX_SHARE:
            warnings.append(f"{category}: repeated reference prefix share looks high ({metrics['top_ref_prefix_share']:.2f})")

    if hard_examples:
        hard_errors.append(f"hard-fail examples found: {len(hard_examples)}")

    report = render_report(
        reference_path,
        candidate_path,
        hard_errors,
        warnings,
        hard_examples,
        reference_metrics,
        candidate_metrics,
        strict_v3,
    )
    return hard_errors, warnings, report


def strict_category_findings(row):
    hard = []
    warnings = []
    category = row.get("category")
    questions = row.get("questions") if isinstance(row.get("questions"), list) else []
    references = row.get("references") if isinstance(row.get("references"), list) else []
    q_text = "\n".join(text for text in questions if isinstance(text, str))
    r_text = "\n".join(text for text in references if isinstance(text, str))

    if category == "글쓰기":
        for label, pattern in STRICT_META_PATTERNS:
            if pattern.search(r_text):
                hard.append(f"{label} in writing reference: {r_text[:120]}")

    if category == "코딩":
        if "```python" not in r_text and "def " not in r_text:
            hard.append("coding reference lacks task-specific Python code")
        if not re.search(r"(assert|print\(|테스트|예시|입력|출력|시간\s*복잡도|공간\s*복잡도|O\()", r_text):
            warnings.append("coding reference may lack example/test or complexity discussion")

    if category == "이해":
        first_question = questions[0] if questions and isinstance(questions[0], str) else ""
        if len(first_question) < 100:
            hard.append("understanding first question is too short for a passage-grounded task")
        has_explicit_question_boundary = bool(re.search(r"\n\s*(질문|문제)\s*:", first_question))
        has_grounding_language = bool(
            re.search(r"(지문|문단|단락|근거|제시|따르면|요약|반론|주장)", q_text + "\n" + r_text)
        )
        if not has_explicit_question_boundary and not has_grounding_language:
            warnings.append("understanding row may not cite or paraphrase passage evidence")

    if category == "문법":
        if not re.search(r"['\"“‘].+['\"”’]", q_text):
            warnings.append("grammar row may lack concrete quoted sentence(s)")
        if QUOTED_CASE_PARTICLE_TOPIC_PATTERN.search(q_text + "\n" + r_text):
            warnings.append("grammar row may use '은' after a quoted case-marked expression; check whether '는' is required")

    return hard, warnings


def group_by_category(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row.get("category")].append(row)
    return grouped


def category_metrics(grouped):
    output = {}
    for category, rows in grouped.items():
        questions = [q for row in rows for q in row.get("questions", []) if isinstance(q, str)]
        refs = [r for row in rows for r in row.get("references", []) if isinstance(r, str)]
        ref_lengths = [len(r) for r in refs]
        prefixes = Counter(normalize(r, strip_numbers=True)[:55] for r in refs)
        output[category] = {
            "rows": len(rows),
            "q_unique_ratio": ratio_unique(questions),
            "ref_unique_ratio": ratio_unique(refs),
            "ref_mean": statistics.mean(ref_lengths) if ref_lengths else 0.0,
            "ref_count": len(refs),
            "top_ref_prefix_share": (prefixes.most_common(1)[0][1] / len(refs)) if refs else 0.0,
        }
    return output


def ratio_unique(values):
    if not values:
        return 0.0
    return len({normalize(value, strip_numbers=True) for value in values}) / len(values)


def render_report(
    reference_path,
    candidate_path,
    hard_errors,
    warnings,
    hard_examples,
    reference_metrics,
    candidate_metrics,
    strict_v3=False,
):
    if hard_errors:
        status = "FAIL"
    elif warnings:
        status = "PASS_WITH_WARNINGS"
    else:
        status = "PASS"

    lines = [
        "# Augmentation Quality Gate Report",
        "",
        f"Status: **{status}**",
        "",
        f"- Reference: `{reference_path}`",
        f"- Candidate: `{candidate_path}`",
        f"- Strict v3 category checks: `{'enabled' if strict_v3 else 'disabled'}`",
        "",
        "## Category Metrics",
        "",
        "| Category | Rows | Q Unique | Ref Unique | Ref Mean | High Ref Mean | Top Ref Prefix |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for category in sorted(candidate_metrics):
        metrics = candidate_metrics[category]
        ref = reference_metrics.get(category, {})
        lines.append(
            "| {category} | {rows} | {q:.2f} | {r:.2f} | {mean:.1f} | {high:.1f} | {prefix:.2f} |".format(
                category=category,
                rows=metrics["rows"],
                q=metrics["q_unique_ratio"],
                r=metrics["ref_unique_ratio"],
                mean=metrics["ref_mean"],
                high=ref.get("ref_mean", 0.0),
                prefix=metrics["top_ref_prefix_share"],
            )
        )

    lines.extend(["", "## Hard Failures", ""])
    if hard_errors:
        lines.extend(f"- {issue}" for issue in hard_errors[:80])
    else:
        lines.append("- None")

    lines.extend(["", "## Warnings", ""])
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings[:80])
    else:
        lines.append("- None")

    lines.extend(["", "## Hard-Fail Examples", ""])
    if hard_examples:
        lines.extend(f"- {example}" for example in hard_examples[:40])
    else:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def run_self_test():
    fenced_code = '```python\nmessage = f"{name}"\n```'
    assert "unfilled placeholder" not in set(forbidden_findings(fenced_code))
    inline_code = 'return `"run:{name}:{count}"` from the callback'
    assert "unfilled placeholder" not in set(forbidden_findings(inline_code))
    assert "unfilled placeholder" in set(forbidden_findings("replace {name} before publishing"))
    set_notation = "정의역을 {a,b}로 두고 연구원={a}, 분석가={a}, 시인={b}로 둔다."
    assert "unfilled placeholder" not in set(forbidden_findings(set_notation))
    assert "numeric condition marker" not in set(forbidden_findings("무차별 조건 4p=3-2p에서 p=1/2이다."))
    assert "numeric condition marker" in set(forbidden_findings("조건 3 다음 문항"))

    understanding_hard, understanding_warnings = strict_category_findings(
        {
            "category": "이해",
            "questions": ["긴 지문 내용이다. " * 12 + "\n\n질문: 핵심 내용을 설명하라.", "답을 보완하라."],
            "references": ["핵심 내용을 설명한다.", "보완 설명이다."],
        }
    )
    assert not understanding_hard, understanding_hard
    assert not any("passage evidence" in warning for warning in understanding_warnings), understanding_warnings

    _, grammar_warnings = strict_category_findings(
        {
            "category": "문법",
            "questions": ["‘동화를’은 어떤 조사 결합 오류인가?", "바르게 고쳐라."],
            "references": ["목적격 조사 뒤에는 보조사 ‘는’을 쓴다.", "‘동화를’는으로 고친다."],
        }
    )
    assert any("quoted case-marked expression" in warning for warning in grammar_warnings), grammar_warnings

    with tempfile.TemporaryDirectory() as tmp:
        ref = Path(tmp) / "ref.jsonl"
        good = Path(tmp) / "good.jsonl"
        warn = Path(tmp) / "warn.jsonl"
        bad = Path(tmp) / "bad.jsonl"
        rows = [
            {
                "id": 1,
                "category": "추론",
                "questions": ["복합 상황 A를 분석하라.", "결론을 영어로 요약하라."],
                "references": ["가정과 이해관계자를 나누어 분석한다. " * 25, "The conclusion balances values. " * 15],
            },
            {
                "id": 2,
                "category": "추론",
                "questions": ["복합 상황 B를 평가하라.", "숫자 근거를 점검하라."],
                "references": ["기준을 세우고 반례를 검토한다. " * 25, "수치와 전제가 일치하는지 확인한다. " * 15],
            },
        ]
        ref.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")
        good_rows = [
            {
                "id": 11,
                "category": "추론",
                "questions": ["도시 정책 C를 비교 기준별로 판단하라.", "그 판단을 짧은 영어 문장으로 바꾸라."],
                "references": [
                    "비용, 형평성, 실행 가능성을 분리해 비교한다. " * 25,
                    "The policy should balance cost and fairness. " * 15,
                ],
            },
            {
                "id": 12,
                "category": "추론",
                "questions": ["조직 의사결정 D의 숨은 전제를 찾아라.", "반대 사례 하나를 들어 보완하라."],
                "references": [
                    "명시된 목표와 실제 제약을 나누어 숨은 전제를 찾는다. " * 25,
                    "반대 사례는 결론의 적용 범위를 좁히는 데 쓰인다. " * 15,
                ],
            },
        ]
        good.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in good_rows), encoding="utf-8")
        warn_rows = [
            {
                "id": 21,
                "category": "추론",
                "questions": ["짧은 정책 판단 C를 하라.", "짧게 보완하라."],
                "references": ["짧은 답.", "짧은 보완."],
            },
            {
                "id": 22,
                "category": "추론",
                "questions": ["짧은 정책 판단 D를 하라.", "짧게 반박하라."],
                "references": ["간단한 답.", "간단한 반박."],
            },
        ]
        warn.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in warn_rows), encoding="utf-8")
        bad_rows = [
            {
                "id": 20,
                "category": "추론",
                "questions": ["후속 검토 20001: 이전 사례와 구별되는 조건 1를 반영하라.", "후속 검토 20002"],
                "references": ["공공성와 운영 재원가 충돌한다. 던/든는 틀린 표기다. " * 20, "{m} placeholder"],
            }
        ]
        bad.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in bad_rows), encoding="utf-8")
        good_errors, good_warnings, _ = validate(ref, good, 2)
        warn_errors, warn_warnings, _ = validate(ref, warn, 2)
        bad_errors, _, _ = validate(ref, bad, 1)
        assert not good_errors, good_errors
        assert not good_warnings, good_warnings
        assert not warn_errors, warn_errors
        assert warn_warnings, "warning candidate should warn"
        assert bad_errors, "bad candidate should fail"
    print("self-test passed")


def main():
    parser = argparse.ArgumentParser(description="Validate LogicKor high-like augmentation candidates.")
    parser.add_argument("--reference", default="data/logickor_sft_high.jsonl")
    parser.add_argument("--candidate", required=False)
    parser.add_argument("--expected-per-category", type=int, default=50)
    parser.add_argument("--report", default=None)
    parser.add_argument("--strict-v3", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return
    if not args.candidate:
        raise SystemExit("--candidate is required unless --self-test is used")

    hard_errors, warnings, report = validate(args.reference, args.candidate, args.expected_per_category, args.strict_v3)
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8")
    print(report)
    raise SystemExit(1 if hard_errors else 0)


if __name__ == "__main__":
    main()
