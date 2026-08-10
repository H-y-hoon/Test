# Spec: High-Like LogicKor Augmentation v3

## Goal

Generate the next accepted LogicKor high-like augmentation set:

- Final file: `data/logickor_sft_high_augmented_v3.jsonl`
- Report: `results/augmentation_quality_report_v3.md`
- Exactly 300 accepted rows, 50 per category
- Schema per row: exactly `id`, `category`, `questions`, `references`
- Each row has 2 questions and 2 references

This run exists to fix the v2/pilot_v2 weaknesses: template collapse, shallow answers, writing-format failures, coding sameness, weak passage grounding, and insufficient math difficulty.

## Inputs

Use as style and quality references:

- `data/logickor_sft_high.jsonl`
- `prompts/logicor_fewshot_augmentation_prompt.md`
- `results/high_vs_augmented_v2_quality_analysis.md`
- `results/pilot_category_expert_review.md`
- `results/augmentation_quality_gate_report_v2.md`

Do not use `data/logickor_sft_middle.jsonl` as a few-shot source. Do not rewrite existing high rows, v2 rows, or pilot rows as new rows.

## Generation Protocol

Generate by category, not as one undifferentiated batch.

1. Read high rows for the target category and infer task shape, difficulty, answer style, and second-turn behavior.
2. Oversample candidates internally: aim for at least 70 candidates per category, accept only 50.
3. Do not persist rejected candidates as durable files.
4. Regenerate hard failures from scratch. Do not patch obviously templated rows.
5. Run the strict validator on the final 300-row file.
6. Send every category to expert review and append the conclusions to the report.
7. Final acceptance requires no rejected category.

## Required Validation

Run exactly this gate before final acceptance:

```bash
python3 scripts/validate_augmented_quality.py \
  --reference data/logickor_sft_high.jsonl \
  --candidate data/logickor_sft_high_augmented_v3.jsonl \
  --expected-per-category 50 \
  --strict-v3 \
  --report results/augmentation_quality_report_v3.md
```

If the command exits nonzero, repair or regenerate and rerun it.

## Global Rejection Rules

Reject any row with:

- generation scaffolding: `이전 사례`, `조건 N`, `후속 검토`, artificial IDs, placeholders
- the same question frame with only nouns or numbers changed
- a second answer that does not depend on the first turn
- source-question copying from high, v2, pilot, or middle files
- awkward particle artifacts such as `도입를`, `공공성와`, `예문를`
- generic filler such as `문제 요구에 맞게 수정하면 된다`

## Category Requirements

### 추론

Limit public-policy/institutional tradeoff prompts to at most 8 of 50. Cover formal logic, fallacy classification, causal/statistical reasoning, probability/Bayes, game theory, puzzle inference, decision theory, hidden assumptions, and counterexample analysis. Remove conditional or quantifier ambiguity before acceptance.

### 수학

Use difficulty bands. Include elementary calculation only as a minority. Cover probability/statistics, functions/limits/calculus, linear algebra, number theory, combinatorics, graph/optimization, proof, and error diagnosis. References must show derivation, not only final answers.

### 글쓰기

Regenerate from scratch. Do not reuse pilot_v2 writing style. The answer must be the requested artifact, not commentary about the artifact. Enforce exact format/count constraints: sentence count, line count, bullet count, tone, audience, word/character limits, and genre. Cover editorial, apology, proposal, dialogue, product copy, speech, review, policy memo, summary rewrite, announcement, and translation/adaptation.

### 코딩

Broaden beyond algorithm interviews. Include debugging, tests, API/data handling, DB/SQL, security edge cases, refactoring, CLI/file handling, runtime behavior, ML/tooling, and standard algorithms. References must include task-specific executable Python code, a short explanation, complexity when relevant, and at least one example or test.

### 이해

Every row must be passage-grounded. Each row needs a distinct short passage and questions that require evidence from that passage. Cover main claim, contradiction, inference, evidence location, table/classification, title selection, implication, and author intent. Generic concept explanation is rejected.

### 문법

Every row must include concrete problematic sentence(s), correction(s), the governing rule, and all requested subanswers. Avoid malformed slash-pair artifacts such as `던/든는`, `안/않는`, `왠/웬는`. Do not force 순화 when the original expression is already acceptable.

## Expert Review Contract

Append this section to `results/augmentation_quality_report_v3.md` after the automatic gate:

- One verdict per category: `PASS`, `CONDITIONAL`, or `REJECT`
- Quantitative checks: counts, duplicates, source-copy check, length/diversity stats
- Qualitative checks: high-set similarity, natural Korean, task difficulty, answer correctness, template risk
- Repair notes for any conditional category

Do not mark v3 accepted if any category is `REJECT`.

## Execution Note

Run from an attached tmux session with 3 workers unless the layout comfortably supports more panes. Keep final outputs to the v3 JSONL and v3 report.
