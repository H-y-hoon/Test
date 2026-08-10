# LogicKor Few-shot Contamination 대응 정리

## 결론

현재 `data/logickor_sft_high.jsonl`에 LogicKor 평가셋의 실제 Q-A에서 유래한 샘플이 포함되어 있고, 이 파일을 same-category few-shot/reference source로 사용한다면 contamination으로 보는 것이 맞다.

핵심 문제는 few-shot 자체가 아니다. 문제는 **평가셋 실제 Q-A 또는 그 변형이 few-shot source에 들어가 학습 데이터 생성에 영향을 주는 것**이다.

따라서 현재 high set을 그대로 positive reference로 쓰는 파이프라인은 clean setting으로 주장하기 어렵다. 파이프라인 구조는 유지하되 few-shot source를 clean seed로 교체해야 한다.

## 왜 Contamination인가

LogicKor 평가셋의 실제 질문, 정답, 풀이, 또는 그 변형이 high set에 들어가고, 이후 증강 과정에서 high set이 few-shot으로 다시 사용되면 평가셋 정보가 다음 경로로 전파된다.

```text
LogicKor eval Q-A
  -> logickor_sft_high.jsonl
  -> same-category few-shot/reference
  -> newly generated SFT data
  -> trained model
  -> LogicKor evaluation
```

이 경우 모델의 LogicKor 점수는 독립적인 일반화 성능으로 보기 어렵다. 설령 새로 생성된 샘플이 평가 문항을 그대로 복사하지 않았더라도, 평가셋의 문제 구조, 답변 방식, reasoning pattern이 generation prompt에 노출되었기 때문이다.

## Few-shot은 유지할 수 있는가

유지할 수 있다. 다만 few-shot source가 평가셋과 독립적이어야 한다.

허용 가능한 source:

- 사람이 새로 작성한 clean seed Q-A
- 평가셋을 보지 않고 만든 zero-shot instruction 기반 seed
- 평가셋과 무관한 공개/자체 reasoning 예시
- 카테고리, 답변 깊이, 문체, 출력 형식만 설명하는 style-only 예시

금지해야 하는 source:

- LogicKor 평가셋 실제 질문
- LogicKor 평가셋 실제 정답 또는 해설
- 평가셋 문항의 paraphrase/rewrite
- 평가셋의 고유 상황, 숫자, 선택지, 소재를 유지한 변형
- contamination 가능성이 있는 기존 `logickor_sft_high.jsonl`

## 권장 파이프라인

```text
clean_seed_examples.jsonl
  -> few-shot / style reference
  -> category expert generation
  -> validator
  -> expert review
  -> bounded repair
  -> overlap audit against LogicKor eval
  -> clean accepted SFT data
```

기존 contaminated high set은 clean 생성 source에서 제외한다.

권장 파일 분리:

```text
data/logickor_sft_high_contaminated.jsonl
data/logickor_clean_seed_examples.jsonl
data/logickor_sft_high_clean_generated.jsonl
```

## Expert Worker 주장의 한계와 역할

Expert worker를 두었다는 사실만으로 contamination이 사라지지는 않는다. 오염된 few-shot을 보고 생성한 데이터를 expert가 검수해도, 평가셋 정보가 generation context에 들어간 사실은 바뀌지 않는다.

따라서 expert worker는 다음처럼 주장해야 한다.

- contamination 제거 장치가 아니라 품질 보증 장치다.
- few-shot 의존도를 줄이기 위한 독립 검수/repair 장치다.
- shallow answer, template collapse, hard failure, warning case를 줄이기 위한 장치다.
- 최종 accept 여부를 few-shot similarity가 아니라 독립 품질 기준으로 판단하기 위한 장치다.

방어 가능한 표현:

> We do not rely on evaluation-derived few-shot examples. Few-shot examples, when used, are limited to clean human-written seeds for style and task-shape conditioning. To reduce dependence on these exemplars, the pipeline uses expert workers for independent review, failure detection, and bounded repair. Thus, few-shot serves as a weak formatting/context signal rather than as the primary source of reasoning content.

피해야 할 표현:

> Expert workers remove contamination from evaluation-derived few-shot examples.

이 표현은 부정확하다. expert worker는 오염된 source를 clean하게 만들지 못한다.

## Clean Setting을 위한 최소 조치

1. `data/logickor_sft_high.jsonl`을 few-shot source로 사용하지 않는다.
2. 평가셋과 독립적인 `clean_seed_examples.jsonl`을 만든다.
3. few-shot은 카테고리, 문체, 답변 깊이, 출력 형식 conditioning에만 사용한다.
4. 생성 결과에 대해 평가셋과 exact/near-duplicate overlap audit을 수행한다.
5. 기존 high 기반 결과는 contaminated result로 분리하거나 main claim에서 제외한다.

## 보고서용 문구

> We found that the previous high-quality synthetic set was generated or reused with few-shot exemplars that included examples derived from actual LogicKor evaluation Q-A pairs. We therefore treat models trained on that version as contaminated and do not use their LogicKor scores as evidence of generalization.
>
> In the clean setting, we retain the same augmentation pipeline but replace the few-shot source with independently constructed clean seed examples. These examples are used only to specify task shape, response style, and reasoning depth. Evaluation-derived questions, answers, rationales, and paraphrases are prohibited from prompts, references, and seed examples. We additionally apply overlap filtering against the LogicKor evaluation set before training.

## 최종 입장

Few-shot 자체는 문제가 아니다. 평가셋 유래 Q-A를 few-shot으로 사용하는 것이 문제다.

현재 파이프라인의 당위성은 다음처럼 정리하는 것이 가장 안전하다.

> The pipeline keeps few-shot conditioning only as a weak style and task-shape signal, while expert workers, validators, and bounded repair reduce dependence on the exemplars and enforce independent quality control. For a clean evaluation claim, all few-shot examples must come from evaluation-independent human-written or zero-shot-generated seeds.
