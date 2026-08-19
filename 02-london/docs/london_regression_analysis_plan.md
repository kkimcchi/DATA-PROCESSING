# 런던 자전거 대여 수요 회귀분석 통합 계획

목표: `01-washington`에서 검증한 범용 회귀분석 파이프라인(STEP 0~13)을 런던 데이터에
그대로 적용해, **2015년 데이터(Train)로 학습한 모델로 2016년(Test) 일별 대여량(`cnt`)을
예측**하고, 가장 일반화 성능이 좋은 모델을 채택한다. 이 문서는 계획안이며, 아직
전처리·모델링 코드는 실행하지 않았다.

## 1. 참고 기준: 01-washington 파이프라인 요약

`01-washington/README.md`, `docs/methodology.md`, `docs/claude_code_workflow.md`를
검토한 결과를 요약한다.

### 1-1. STEP 구성 (STEP 0~13)

| STEP | 내용 |
|---|---|
| 0 | 스키마 상수 정의(`TARGET`/`DATE_COL`/`CATEGORICAL_COLS`/`NUMERIC_COLS`/`DROP_COLS`) |
| 1 | EDA (info/describe/결측치/중복/고유값, 분포 시각화) |
| 2 | 결측치 처리 |
| 3 | 이상치 탐지 및 처리 (IQR 기준, 입력오류가 아니면 함부로 삭제하지 않음) |
| 4 | 날짜 처리 및 파생 변수 |
| 5 | 데이터 누수(leakage) 변수 제거 |
| 6 | 범주형 변수 원-핫 인코딩 (`pd.get_dummies(..., drop_first=True)`) |
| 7 | 다중공선성 점검 (VIF > 10, 보수적으로 5) |
| 8 | 스케일링(Ridge/Lasso 등 페널티 모델용) |
| 9 | 학습/테스트 분할 — **시간순 분할** (무작위 분할 금지) |
| 10~11 | 회귀모델 적합/평가 (OLS/Ridge/Lasso/Random Forest/XGBoost) |
| 12 | 회귀 가정 진단 (잔차, Q-Q, Durbin-Watson, Breusch-Pagan) |
| 13 | 결과 해석 및 리포팅 |

### 1-2. 핵심 교훈 (그대로 계승)

- **전체 데이터 적합 R²만으로 모델을 비교하면 안 됨** — 워싱턴에서 전체 적합 기준
  Random Forest(R²=0.98)가 선형회귀(R²=0.86)보다 우수해 보였지만, 시간순
  train/test로 나누자 순위가 뒤집힘(선형회귀 test R²=0.65 > 기본 RF test R²=0.50).
  **반드시 test 성능으로만 모델을 비교·채택한다.**
- 정규화 선형모델(Ridge/Lasso)도 동일한 방법론(그리드서치)으로 튜닝해 "선형회귀만
  튜닝 안 한 것 아니냐"는 공정성 문제를 실측으로 검증한다.
- 다중공선성 컬럼(예: `workingday` VIF=inf, `atemp` VIF≈650)은 STEP7에서 실측 후
  제거한다 — 미리 짐작하지 않고 계산해서 판단한다.
- 최종 채택 모델: 튜닝 XGBoost (test R²=0.727) — 워싱턴 사례 기준. 런던에서도
  동일한 5개 모델을 같은 방식으로 비교해 **런던 자체 기준으로 최적 모델을 새로
  결정**한다(워싱턴 결과를 그대로 가져다 쓰지 않음).
- 회귀 파이프라인과 별도로 IsolationForest + counterfactual 민감도 분석으로
  "평소와 다른 날"을 탐지·설명하는 보조 EDA를 수행한다(데이터를 수정하지 않고
  탐지·설명만).
- 문서 구조: README(파이프라인+결과) / methodology.md(STEP별 근거) /
  claude_code_workflow.md(작업 기록) 3분할 방식을 런던에도 적용한다.

## 2. 런던 데이터 현황

| 구분 | 파일 | 행 수 | 용도 |
|---|---|---|---|
| Train | `data/processed/london_daily_2015.csv` | 362일 | 학습 |
| Test | `data/processed/london_daily_2016.csv` | 365일 | 평가(예측 대상) |

컬럼: `instant, dteday, yr, mnth, weekday, is_holiday, is_weekend, workingday, season,
weather_code, t1, t2, hum, wind_speed, cnt, n_hours_recorded`

(위 두 파일과 파생 규칙은 `docs/london_hourly_to_daily_conversion_plan.md`,
`docs/hourly_to_daily_aggregation_rules.md` 참고 — 이미 생성 완료된 산출물이다.)

## 3. STEP 0: 런던 스키마 상수(안)

| 상수 | 워싱턴 | 런던(안) | 변경 사유 |
|---|---|---|---|
| `TARGET` | `cnt` | `cnt` | 동일 |
| `DATE_COL` | `dteday` | `dteday` | 동일 |
| `CATEGORICAL_COLS` | `season, yr, mnth, holiday, weekday, workingday, weathersit` | `season, mnth, weekday, is_holiday, workingday, weather_code` | `yr` 제외(4절 참고), `holiday`→`is_holiday`, `weathersit`→`weather_code` 이름만 대응 |
| `NUMERIC_COLS` | `temp, atemp, hum, windspeed` | `t1, t2, hum, wind_speed` | 이름만 대응 (스케일은 실측값으로 다름 — 규칙 적용엔 무관) |
| `DROP_COLS` (식별자/누수 변수, 완전 제거) | `instant, casual, registered` | `instant` | 런던엔 `casual`/`registered` 자체가 없어 별도 누수 변수 제거가 필요 없음 |
| 모델 입력에서 제외(컬럼은 유지, 메타데이터 취급) | 해당 없음 | `yr, n_hours_recorded, is_weekend` | 4절 참고 — 삭제가 아니라 "입력 변수로 안 씀" |

## 4. 워싱턴과 다른 지점 — 실행 전 확정이 필요한 결정 사항

런던 데이터를 실제로 읽어 확인한 결과, 워싱턴에는 없었던 아래 이슈들이 있다. 각
항목에 제안을 달아두되, 실행 단계에서 실측(VIF 등)으로 재확인 후 최종 확정한다.

1. **`n_hours_recorded < 24`인 날 — Train 18일 / Test 16일**
   `cnt`는 합계 집계라 이 날들은 "부분 합"이다(과소집계). 규칙서 원칙대로 날짜를
   버리지 않고 유지하되, `n_hours_recorded`는 모델 입력 변수에서 제외하고
   잔차 분석 시 "오차가 큰 날이 실제로 이 34일에 몰리는지" 확인하는 용도로만 쓴다.
   (만약 왜곡이 크다고 판단되면 이 날들을 제외한 버전도 비교해볼 수 있음 — 지금은
   기본안만 확정하고, 필요 시 실행 단계에서 추가 실험으로 다룬다.)

2. **`yr` 컬럼 — Train은 전부 2015, Test는 전부 2016**
   분할 자체가 연도 경계이므로 각 분할 내에서 `yr`은 분산이 0인 상수 컬럼이다.
   학습에 아무 정보가 없으므로 모델 입력에서 제외한다.

3. **`weekday` / `is_weekend` / `workingday` / `is_holiday` 간 결정론적 관계**
   `is_weekend`는 `weekday`에서, `workingday`는 `is_holiday`+`is_weekend`에서
   그대로 계산된 값이라 서로 완전히 종속적이다. 워싱턴의 `workingday`(VIF=inf) 사례와
   같은 패턴이 재현될 가능성이 높다. **미리 제거하지 않고 STEP7 VIF 점검에서 실측
   후 제거 대상을 확정한다** (워싱턴과 동일한 방식 — 짐작하지 않고 계산).

4. **`t1` ↔ `t2` 상관계수 0.992 (Train 기준 실측)**
   워싱턴의 `temp`↔`atemp`(VIF≈650) 사례와 동일한 패턴. STEP7 VIF 점검에서 제거
   후보로 다룬다(짐작이 아니라 실측 VIF로 판단).

5. **`weather_code=26`(눈)이 Train(2015)엔 0건, Test(2016)엔 1건만 존재**
   원-핫 인코딩을 Train 기준으로 `fit`한 뒤, Test는 Train과 동일한 컬럼셋으로
   `reindex(fill_value=0)`해야 한다(안 그러면 Test에만 있는 범주 때문에 컬럼 수가
   달라져 예측이 깨짐). 1건뿐이라 성능에 미치는 영향은 미미할 것으로 예상되지만,
   파이프라인이 에러 없이 이 케이스를 처리하도록 명시해둔다.

6. **`weather_code`/`season`은 원본 코드 값 그대로 사용** (재매핑 없음). 다만
   `weather_code`의 일별 대표값(최빈값) 규칙은 런던에 대조용 일별 정답 파일이 없어
   검증 불가능한 근사 규칙이라는 캐비앗이 있음(`hourly_to_daily_aggregation_rules.md`
   참고) — 회귀 결과 해석 시 이 컬럼의 신뢰도가 다른 컬럼보다 낮다는 점을 감안한다.

7. **관측 기간이 2개 연도(2015/2016)뿐, 그마저도 연도 경계로 train/test를 나눔**
   워싱턴은 731일(2개년) 전체를 함께 학습해 계절 패턴을 두 번 관측한 뒤 무작위가
   아닌 시간순으로 뒤쪽 20%만 떼어 테스트했다. 런던은 아예 "2015년 전체로 학습 →
   2016년 전체로 예측"이므로, 모델이 **한 번도 본 적 없는 해**를 예측해야 한다.
   연도 효과와 계절 효과가 완전히 분리되지 않는 구조적 한계가 있음을 최종 리포트에
   명시한다(예: 2016년에만 있었던 이례적 날씨가 있다면 모델이 못 배운 패턴일 수 있음).

## 5. Train/Test 분할

| | 워싱턴 | 런던 |
|---|---|---|
| 방식 | 시간순 80/20 (앞 585일 학습 / 뒤 146일 테스트) | **연도 단위 고정 분할** (요청사항) |
| Train | 2011-01-01 ~ (585일) | 2015년 전체 (362일) |
| Test | (146일) ~ 2012-12-31 | 2016년 전체 (365일) |
| 제외 | 없음 | 2017년 3일 (2017-01-01~03, 이미 분할 완료) |

하이퍼파라미터 튜닝 시 교차검증(`TimeSeriesSplit`)은 **Train(2015) 내부에서만**
수행하고, Test(2016)는 최종 평가에만 사용한다(워싱턴과 동일 원칙 — 데이터 누수 방지).

## 6. 비교할 모델 및 평가

- 모델: OLS(선형회귀), Ridge, Lasso, Random Forest, XGBoost — 워싱턴과 동일 5종
- 지표: R², RMSE, MAE — **Test(2016) 기준으로만 모델을 비교·채택**(1-2절 교훈)
- 절차: 기본 모델 5종 우선 비교 → Ridge/Lasso/RF/XGBoost 하이퍼파라미터 튜닝
  (`TimeSeriesSplit` 그리드서치, Train 내부) → 튜닝 전/후 비교 → 최종 모델 채택
- 최종 채택 모델로 2016년 366일치(실제 365일) `cnt` 예측값 산출, 실제값과 비교한
  잔차 분석(특히 4절 1번의 `n_hours_recorded<24`인 16일에 오차가 몰리는지 확인)

## 7. 보조 EDA: 이상치 탐지 (회귀 파이프라인과 별개)

워싱턴과 동일한 방법(IsolationForest, `n_estimators=300, contamination=0.05` +
counterfactual 민감도 분석으로 원인 변수 판정)을 런던에 적용한다.

- 특성: `t1, t2, hum, wind_speed, weather_code, cnt` (워싱턴의 6개 특성에 대응)
- 데이터는 수정하지 않고 탐지·설명만 수행
- 2015+2016 전체(730일) 대상으로 실행할지, Train(2015)만 대상으로 할지는 실행
  단계에서 결정 — 이상 탐지는 예측 모델이 아니므로 데이터 누수 문제는 없어
  전체 기간을 함께 봐도 무방함 (제안: 730일 전체로 실행)

## 8. 산출물 계획 (폴더 구조)

```
02-london/
  src/
    hourly_to_daily.py            (완료)
    split_daily_by_year.py        (완료)
    preprocessing.py              (예정 — STEP 0~9, 4절 결정사항 반영)
    regression.py                 (예정 — OLS, build_design_matrix)
    random_forest.py              (예정)
    xgboost_model.py              (예정)
    train_test_eval.py            (예정 — 연도 기준 분할 버전)
    tune_linear.py                (예정)
    tune_random_forest.py         (예정)
    tune_xgboost.py               (예정)
    anomaly_detection.py          (예정)
    plot_anomalies.py             (예정)
  data/processed/
    london_daily.csv              (완료)
    london_daily_2015.csv         (완료, Train)
    london_daily_2016.csv         (완료, Test)
    london_daily_preprocessed_*.csv (예정)
    london_daily_anomalies.csv    (예정)
  figures/                        (예정)
  docs/
    hourly_to_daily_aggregation_rules.md      (완료)
    london_hourly_to_daily_conversion_plan.md (완료)
    london_regression_analysis_plan.md        (본 문서)
    london_methodology.md                     (예정 — 워싱턴 methodology.md 대응)
    README.md 결과 절 갱신 또는 london_final_report.md (예정)
```

## 9. 실행 로드맵

1. 본 계획 승인
2. 전처리 스크립트 작성·실행 (STEP 0~9, 4절 결정사항 반영) → EDA로 재확인
3. STEP 7 다중공선성 점검 → `t1`/`t2`, `weekday`/`is_weekend`/`workingday` 중
   실제 제거 대상 실측 확정
4. 5개 모델 기본 학습(Train=2015) → Test(2016) 평가 → 결과표 작성
5. Ridge/Lasso/Random Forest/XGBoost 하이퍼파라미터 튜닝
6. 최종 모델로 2016년 `cnt` 예측 + 잔차 분석
7. 이상치 탐지(보조 EDA) 실행
8. 최종 리포트 작성 (워싱턴 README/methodology 구조에 대응)

승인 및 4절 결정사항에 이견 없으면, 다음 단계로 전처리 스크립트부터 순서대로
작성·실행하겠습니다.
