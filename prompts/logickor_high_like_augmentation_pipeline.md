# Spec: High-Like LogicKor Augmentation Pipeline v2

## Goal

Regenerate a stricter pilot that is closer to `data/logickor_sft_high.jsonl` than to the rejected template-like v2 output and the rejected first pilot. Do not scale to the final 300 rows until this pilot passes both the automatic gate and category-expert review.

## Inputs

- High reference only: `data/logickor_sft_high.jsonl`
- Prompt context: `prompts/logicor_fewshot_augmentation_prompt.md`
- Failure analysis: `results/high_vs_augmented_v2_quality_analysis.md`
- Pilot expert review: `results/pilot_category_expert_review.md`

Do not use `data/logickor_sft_middle.jsonl` as few-shot source.

## Outputs

- Pilot v2: `data/logickor_sft_high_augmented_pilot_v2.jsonl` with exactly 10 accepted rows per category
- Pilot v2 report: `results/augmentation_quality_gate_report_v2.md`
- Final: `data/logickor_sft_high_augmented_v3.jsonl` with 50 accepted rows per category

Do not overwrite the first pilot or existing dataset files. Do not keep failed candidates as durable outputs.

## Required Gate

Run this before treating the pilot as accepted:

```bash
python3 scripts/validate_augmented_quality.py \
  --reference data/logickor_sft_high.jsonl \
  --candidate data/logickor_sft_high_augmented_pilot_v2.jsonl \
  --expected-per-category 10 \
  --report results/augmentation_quality_gate_report_v2.md
```

The command must exit with status 0, meaning there are no hard failures. Warnings are not automatic rejection; send them to category-expert review.

## Batch Protocol

1. Read high rows by category and infer task variety, difficulty, answer style, and second-turn behavior.
2. Generate more than 10 candidates per category, but save only accepted rows.
3. Regenerate hard failures from scratch; do not patch a templated row into shape.
4. Run the required gate on the pilot v2 file.
5. Send every category to expert review for task diversity, answer correctness, and natural Korean.
6. Regenerate expert-rejected categories from scratch.
7. Scale to 50 rows per category only after the pilot passes hard checks and expert review.

## Global Hard Rejection Rules

Reject rows that contain generation scaffolding such as `이전 사례`, `조건 N`, `후속 검토 20001`, placeholders like `{m}`, or mechanical topic swaps. Reject rows that copy high-set questions, reuse the same question frame with only nouns changed, fail to answer the second question, or contain obvious Korean particle artifacts.

## Category-Specific Requirements

### 추론

At most 2 rows may be public-policy or institutional tradeoff prompts. The category must include at least 6 distinct reasoning families among formal logic, fallacy classification, causal/statistical reasoning, probability/Bayes, game theory, puzzle-style inference, decision theory, and hidden-assumption analysis. Vary the second turn: counterexample, assumption change, English summary, validity check, table, or rule extraction.

### 수학

At most 2 rows may be elementary arithmetic or simple condition-change problems. Include higher-level domains such as probability/statistics, functions/limits/calculus, linear algebra, number theory, combinatorics, graph/optimization, proof, error diagnosis, and concept comparison. References must show derivation, not only a final answer.

### 글쓰기

No more than 2 rows may be 안내문/공지문 style. Cover distinct genres such as editorial, apology, proposal, dialogue, product copy, speech, review, policy memo, and summary rewrite. Every reference must obey requested counts and formats exactly, including sentence counts, bullet counts, tone, and audience.

### 코딩

Every row must require a different algorithmic shape. Do not reuse one `solve(items)` skeleton. References must include executable, task-specific Python code, a short explanation, complexity, and at least one example or test. Reject placeholder advice such as "문제 요구에 맞게 수정하면 된다."

### 이해

Every row must be passage-grounded reading comprehension. Each row needs a distinct short passage and questions that require evidence from that passage. Do not generate generic "개념을 설명하라" tasks. References must cite or paraphrase passage clues.

### 문법

Every row must include concrete problematic sentence(s), correction(s), the governing rule, and an answer to every subquestion. Reject malformed slash-pair artifacts such as `던/든는`, `안/않는`, or `왠/웬는`.

## Expert Review Contract

Append category-expert conclusions to `results/augmentation_quality_gate_report_v2.md` after the automatic gate. Use one verdict per category: `PASS`, `CONDITIONAL`, or `REJECT`. Do not produce the final 300-row dataset unless no category is rejected.

## Warning Review Rules

Treat low normalized diversity, repeated answer prefixes, and short answer length as review warnings, not automatic rejection. Category experts must inspect those rows directly and decide whether the issue is real template collapse or an acceptable high-style pattern.

## Execution Note

Run the project team workflow from an attached tmux session using this file as the binding brief. Use 3 workers unless the tmux layout has enough pane space for more.
