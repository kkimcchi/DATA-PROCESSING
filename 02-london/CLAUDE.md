# CLAUDE.md — 02-london 작업 규칙

이 폴더(`02-london/`)에서 작업할 때 지켜야 할 규칙이다. `01-washington`에는 이
파일이 실제로 존재하지 않지만(`docs/claude_code_workflow.md`에 설명만 있고
최종 배포 저장소엔 포함 안 됨), 이 프로젝트에서는 사용자가 명시적으로 요청해
만들었다.

## 코드를 쓰기 전에

- 회귀분석 관련 작업이면 항상 `docs/london_methodology.md`의 해당 STEP을 먼저
  읽는다. 전처리 관련이면 `docs/london_hourly_to_daily_conversion_plan.md`와
  `docs/hourly_to_daily_aggregation_rules.md`를 먼저 읽는다.
- **워싱턴(`01-washington`)의 결론(DROP_COLS, 최적 모델, VIF 제거 컬럼 등)을
  런던에 그대로 복사하지 않는다.** 컬럼 이름이 같아도(`season`, `weekday` 등)
  값의 실제 관계는 다를 수 있다 — 반드시 런던 데이터로 다시 계산해서 확인한 뒤
  결론을 내린다. (실제로 런던에서 `season`이 `mnth`와 100% 종속인 것은 워싱턴
  에는 없던 문제였고, 최종 우승 모델도 워싱턴(XGBoost)과 정반대(Ridge)였다.)
- 비일반적인 판단(컬럼 제거, 결측 처리 방식, 이상치 처리 등)이 필요하면 먼저
  근거와 제안을 제시하고, 사용자 확인을 받은 뒤 코드에 반영한다. 특히 "아직
  실행하지 마"라는 지시가 있으면 계획 문서 작성까지만 하고 실행 스크립트는
  쓰지 않는다.

## 폴더 구조

```
data/
  london_merged.csv          # 원본 시간별 데이터 — 절대 수정하지 않음(읽기 전용)
  processed/                 # 파생 산출물 (일별/연도별 변환, 예측, 이상치 결과)
  raw/                       # 워싱턴 참고용 원본(day.csv, bike_sharing_hour.csv) — 이것도 읽기 전용
src/                          # 실행 스크립트, 아래 "코딩 컨벤션" 참고
docs/                         # 계획 문서·방법론 문서 (아래 "문서화 원칙" 참고)
figures/                      # PNG 시각화 결과
logs/                         # 작업로그, 아래 "작업로그 관리" 참고
```

## 원본 데이터 보존 원칙

`data/london_merged.csv`, `data/processed/london_daily*.csv`, `data/raw/*.csv`는
모두 읽기 전용으로만 다룬다. 새 파생 결과가 필요하면 `data/processed/`에 새
파일로 저장하고, 기존 파일을 덮어쓰지 않는다.

## 코딩 컨벤션

- 경로는 항상 `BASE_DIR = Path(__file__).resolve().parent.parent`로 절대경로를
  잡는다 — 프로젝트 루트(`02-london/`)든 `src/`든 어디서 실행해도 동작해야 한다.
- 여러 스크립트가 공유하는 로직은 새로 구현하지 말고 재사용한다:
  `regression.py`의 `load_train_test()`/`build_design_matrix()`,
  `train_test_eval.py`의 `evaluate()`를 import해서 쓴다.
- Train/Test를 각각 따로 `pd.get_dummies()`하지 않는다 — 한쪽에만 있는 범주
  (예: `weather_code=26`)가 있으면 컬럼 수가 어긋난다. 항상 `build_design_matrix()`
  처럼 합쳐서 인코딩한 뒤 분리한다.
- 새 파생 변수나 컬럼 제거를 코드에 반영하기 전에, VIF/상관관계처럼 계산 가능한
  근거는 실제로 계산해서 주석에 수치를 남긴다("아마 이럴 것"으로 짐작해서 넣지
  않는다).
- Windows 콘솔에서 한글 출력 스크립트를 실행할 때 `UnicodeEncodeError`(cp949가
  em dash 등을 못 씀)가 날 수 있다 — `PYTHONIOENCODING=utf-8 python xxx.py`로
  실행한다.

## 문서화 원칙

- 국면이 바뀌는 작업(변환 규칙 확정, 새 분석 착수 등) 전에는 계획 문서를 먼저
  `docs/`에 작성하고 승인을 받은 뒤 실행한다 — 계획과 실행을 같은 턴에 섞지 않는다.
- 결과·비교표는 `README.md`, STEP별 실행 근거·중간 계산값은
  `docs/london_methodology.md`, 실행 전 결정 사항은 계획 문서
  (`docs/london_*_plan.md`)에 나눠서 쓴다 — 세 문서의 역할을 섞지 않는다.
- 워싱턴과 다른 판단을 내렸다면(컬럼 처리, 모델 채택 등) 왜 다른지 근거를 함께
  남긴다.

## 작업로그 관리

`logs/{작업일자}_london_작업로그.md` 형식으로 남긴다(형식은
`docs/claude_code_workflow.md` 참고). 파일 안에서 두 섹션을 분리한다:

- `🔖 현재 진행 상태`: 매번 덮어써서 최신 상태만 유지 (완료 항목, 최종 채택 모델,
  남은 과제)
- `작업 로그 (시간순)`: append-only, 새 작업 단위가 끝날 때마다 맨 아래에 추가
  (기존 항목은 수정하지 않음)

세션 도중 실시간으로 갱신하는 것이 원칙이지만, 그러지 못했다면(예: 세션 막판에
한 번에 정리) 그 사실을 로그 파일에 명시한다 — 실시간 기록과 사후 재구성은
신뢰도가 다르므로 구분한다.

## 금지 사항

- 원본 CSV(`london_merged.csv` 등)를 스크립트로 덮어쓰거나 값 수정 금지.
- 계산 없이 워싱턴 결과를 그대로 가져와 런던 결론으로 쓰는 것 금지.
- 사용자가 "아직 실행하지 마"라고 한 뒤 같은 턴/다음 턴에 바로 실행 스크립트를
  작성·실행하는 것 금지 — 계획 문서 승인까지 기다린다.
- 이유 없이 워싱턴 `requirements.txt`를 그대로 복사하는 것 금지 — 실제로 import된
  패키지만 확인해서 담는다(런던은 `seaborn`을 안 써서 뺐다).
