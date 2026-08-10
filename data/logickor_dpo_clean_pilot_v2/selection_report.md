# LogicKor DPO clean subset 선택 보고서

## 선택 원칙

- 세 arm은 동일하게 카테고리당 25개를 사용한다.
- control_balanced: 전체 reviewed pair에서 층화 무작위 추출
- high_balanced: control의 high pair는 유지하고 medium pair만 같은 카테고리의 high pair로 교체
- 두 주 arm은 카테고리 수와 전체 accept/swap 수가 같고, 공통 표본을 최대한 유지한다.
- high_length_priority_balanced: high-confidence 중 카테고리별 token 길이 비가 가까운 순서로 추출
- 길이 우선 arm은 source pool 내 최선의 진단용 subset이며 strict length-matched 데이터가 아니다.

## Source 및 subset 요약

| Arm | N | High | Medium | Accept | Swap | Chosen tok | Rejected tok | Ratio p50 | Ratio p90 | Ratio max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| source | 380 | 301 | 79 | 325 | 55 | 267.16 | 513.09 | 3.252 | 6.431 | 14.944 |
| control_balanced | 150 | 113 | 37 | 128 | 22 | 289.96 | 578.59 | 3.730 | 7.031 | 14.944 |
| high_balanced | 150 | 150 | 0 | 128 | 22 | 279.05 | 543.19 | 3.502 | 7.314 | 14.944 |
| high_length_priority_balanced | 150 | 150 | 0 | 139 | 11 | 255.71 | 459.14 | 1.931 | 4.786 | 6.467 |

## Strict length balance feasibility

| Max token ratio | High rows | Balanced rows/category | 추론 | 수학 | 코딩 | 이해 | 글쓰기 | 문법 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.20 | 31 | 1 | 1 | 1 | 5 | 3 | 20 | 1 |
| 1.33 | 49 | 1 | 1 | 2 | 12 | 4 | 29 | 1 |
| 1.50 | 58 | 1 | 2 | 3 | 13 | 5 | 34 | 1 |
| 2.00 | 98 | 4 | 4 | 7 | 21 | 15 | 45 | 6 |
| 2.50 | 128 | 4 | 4 | 10 | 28 | 20 | 52 | 14 |
| 3.00 | 149 | 5 | 5 | 11 | 29 | 27 | 56 | 21 |

## 판단

- token ratio 1.2 이하 high pair만으로는 6개 카테고리 균형 실험을 만들 수 없다.
- 첫 pilot은 high_balanced를 주 arm으로 사용한다.
- control_balanced는 confidence 효과 비교용이다.
- 두 주 arm은 113개를 공유하고 37개만 교체한다.
- high_length_priority_balanced는 길이 confound의 방향을 보는 보조 arm이다.
- strict length-matched 실험은 기존 380개 필터링만으로 불가능하다.
