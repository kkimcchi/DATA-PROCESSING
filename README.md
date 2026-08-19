# Claude Code로 진행한 자전거 대여 수요 회귀분석 포트폴리오

**"AI가 대신 분석해줬다"가 아니라, Claude Code를 분석 파트너로 통제·검증하며
데이터 분석 워크플로우 전체(확인 → 정제 → 탐색 → 모델링 → 검증 → 문서화)를
끝까지 수행한 기록이다.** 같은 회귀분석 파이프라인을 서로 다른 두 도시의
자전거 공유 데이터셋에 적용해, **재사용 가능한 방법론을 설계하는 능력**과
**새 데이터에서 이전 결론을 맹신하지 않고 매번 재검증하는 태도**를 함께
보여준다.

## 이 포트폴리오가 증명하는 것

| | |
|---|---|
| **재사용 가능한 파이프라인 설계** | STEP 0의 스키마 상수 5개(`TARGET`/`DATE_COL`/`CATEGORICAL_COLS`/`NUMERIC_COLS`/`DROP_COLS`)만 바꾸면 워싱턴에서 만든 전처리·모델링 코드가 런던 데이터에도 그대로 재사용됐다. |
| **짐작이 아니라 실측** | "워싱턴에서 `workingday`가 VIF=inf였으니 런던도 그럴 것"이라고 가정하지 않고, 런던 Train 데이터로 VIF를 처음부터 다시 계산했다. 그 결과 워싱턴에는 없던 새 문제(`season`이 `mnth`로 100% 결정됨)를 실제로 찾아냈다. |
| **모델 우승자를 미리 정해두지 않음** | 워싱턴은 튜닝 XGBoost(R²=0.727)가 우승했지만, 런던에서 같은 5개 모델·8개 설정을 처음부터 다시 비교한 결과 **정반대로 Ridge(선형모델)가 우승**했다. 그 원인(작은 학습 표본에서 트리 모델 내부 교차검증이 불안정, 실측 CV R²=-0.15)까지 실측으로 규명했다. |
| **단계별 계획 → 승인 → 실행** | "아직 실행하지 마", "신뢰도 지표는 만들지 마" 같은 사용자 지시에 따라 계획 문서를 먼저 쓰고 승인받은 뒤에야 코드를 작성·실행하는 패턴을 반복했다 — 즉흥적으로 전체를 밀어붙이지 않았다. |
| **실패도 정직하게 기록** | 성능을 개선하지 못한 실험(`holiday_buffer` 변수, 워싱턴)과 계획에 있었지만 뒤늦게 빠졌던 검증(Breusch-Pagan, 런던)을 숨기지 않고 문서에 남기고 나중에 채워 넣었다. |

## 프로젝트 구성

| | 01-washington | 02-london |
|---|---|---|
| 역할 | **파이프라인 원본 개발** — STEP 0~13 범용 회귀분석 방법론을 처음부터 설계 | **파이프라인 재사용 검증** — 같은 방법론을 구조가 다른 새 데이터셋에 적용 |
| 데이터 | UCI Bike Sharing Dataset (`day.csv`, 731일) | Kaggle London Bike Sharing (시간별 → 일별 자체 변환, 730일) |
| 분할 방식 | 시간순 80/20 (585일 학습 / 146일 테스트) | 연도 단위 고정 분할 (2015년 학습 / 2016년 테스트, 요청사항) |
| 최고 모델 | 튜닝 XGBoost, Test R²=**0.727** | 튜닝 Ridge, Test R²=**0.7713** |
| 핵심 발견 | 전체 데이터 적합 R²만으로 모델을 비교하면 안 됨(과적합 함정 실측 — RF 전체적합 R²=0.98 vs test R²=0.50) | 학습 표본이 작으면 트리 앙상블보다 선형모델이 더 안정적으로 일반화함(내부 CV로 실측) |
| 보조 EDA | IsolationForest 이상치 탐지 — 37일(5.1%), 실제 기상 이벤트(2011 Snowtober 등)와 대조 검증 | 동일 방법 적용 — 37일(5.1%), 날씨코드가 원인 1위(43.2%) |
| 문서 | [`01-washington/README.md`](01-washington/README.md) · [`methodology.md`](01-washington/docs/methodology.md) · [`claude_code_workflow.md`](01-washington/docs/claude_code_workflow.md) | [`02-london/README.md`](02-london/README.md) · [`london_methodology.md`](02-london/docs/london_methodology.md) · [`claude_code_workflow.md`](02-london/docs/claude_code_workflow.md) |

## Claude Code를 이렇게 활용했다

1. **방법론을 코드보다 먼저 문서화** — "왜 이 단계가 필요한가"를 STEP 0~13으로
   정리해두고, 코드는 그 문서를 따라가게 했다. 새 데이터셋(런던)을 만났을 때도
   "다음 데이터셋도 같은 방식으로 해줘"라는 요청 하나로 동일한 절차가 재현됐다.
2. **판단이 필요한 지점마다 먼저 확인** — 결측치 처리 방식, 다중공선성 제거
   대상, 모델 채택 기준처럼 정답이 하나가 아닌 결정은 Claude Code가 먼저
   근거와 후보안을 제시하고, 사람이 확정한 뒤에야 코드에 반영했다.
3. **사용자의 지적을 실측으로 답함** — "선형회귀만 튜닝을 안 한 것 아니냐"는
   공정성 지적에는 Ridge/Lasso를 동일 방법론으로 추가 튜닝해 실측으로 답했고,
   "전체 적합 R²로 비교해도 되냐"는 우려에는 시간순 train/test 재평가로 실제
   순위가 뒤집히는 것을 보여줬다.
4. **문서 역할을 분리해서 관리** — 결과·비교표(README) / STEP별 실행 근거
   (methodology.md) / 실행 전 결정 사항(계획 문서) / 협업 기록(claude_code_workflow.md)
   / 진행 상태(logs)를 서로 다른 파일에 나눠, 대화가 끊겨도 어느 문서를 보면
   되는지 명확하게 했다.

## 기술 스택

`pandas` · `numpy` · `scikit-learn` (Ridge/Lasso/RandomForest, GridSearchCV,
TimeSeriesSplit, IsolationForest) · `statsmodels` (OLS, VIF, Breusch-Pagan,
Durbin-Watson) · `xgboost` · `matplotlib`

## 폴더 구조

```
0819_20cha/
├── README.md                        # 포트폴리오 소개 (본 문서)
├── 01-washington/                   # 프로젝트 1: 파이프라인 원본 개발
│   ├── README.md                    # 결과 리포트 (모델 비교표, 이상치 탐지)
│   ├── docs/
│   │   ├── methodology.md           # STEP 0~13 범용 방법론
│   │   └── claude_code_workflow.md  # Claude Code 협업 기록
│   ├── src/                         # 전처리 · 5개 모델 · 튜닝 3종 · 이상치 탐지
│   ├── data/{raw,processed}/
│   ├── figures/                     # EDA·이상치 시각화 PNG
│   └── requirements.txt
├── 02-london/                       # 프로젝트 2: 파이프라인을 새 데이터셋에 적용
│   ├── README.md                    # 결과 리포트
│   ├── CLAUDE.md                    # 작업 규칙 (원본 보존, 재검증 원칙 등)
│   ├── docs/
│   │   ├── hourly_to_daily_aggregation_rules.md      # 시간별→일별 집계 규칙
│   │   ├── london_hourly_to_daily_conversion_plan.md # 변환 계획서
│   │   ├── london_regression_analysis_plan.md        # 회귀분석 통합 계획
│   │   ├── london_methodology.md                     # STEP 0~13 적용 기록
│   │   └── claude_code_workflow.md                   # Claude Code 협업 기록
│   ├── src/                         # 변환 · 전처리 · 5개 모델 · 튜닝 · 예측 · 이상치 탐지
│   ├── data/{raw,processed}/
│   ├── figures/
│   ├── logs/                        # 작업로그 (진행 상태 + 시간순 기록)
│   └── requirements.txt
└── .gitignore
```

## 시작하기

```bash
cd 01-washington && pip install -r requirements.txt && cd src && python preprocessing.py
cd 02-london      && pip install -r requirements.txt && cd src && python regression.py
```

각 프로젝트의 전체 실행 순서는 해당 폴더 README.md의 "실행" 절 참고.
