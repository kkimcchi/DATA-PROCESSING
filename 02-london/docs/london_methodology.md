# 분석 방법론: 런던 자전거 대여 수요 회귀분석 (STEP 0~13 적용 기록)

`01-washington/docs/methodology.md`의 범용 STEP 0~13 파이프라인을 런던 데이터
(`london_merged.csv` → 일별 집계 → Train=2015/Test=2016)에 실제로 적용한 기록이다.
워싱턴 문서가 "다른 데이터셋에도 재사용 가능한 일반 원칙"을 설명한다면, 이 문서는
그 원칙을 런던에 적용했을 때 **실측으로 무엇이 나왔는지**를 STEP별로 정리한다.
결과 요약·결과표는 [`../README.md`](../README.md), 실행 전 결정 사항은
[`london_regression_analysis_plan.md`](london_regression_analysis_plan.md)를 참고.

## STEP 0. 데이터셋 파라미터 식별

`src/regression.py`에서 확정한 최종 스키마:

```python
TARGET = "cnt"
DATE_COL = "dteday"
NOMINAL_COLS = ["mnth", "weekday", "weather_code"]   # 원-핫 인코딩
BINARY_NUMERIC_COLS = ["is_holiday"]
CONTINUOUS_COLS = ["t1", "hum", "wind_speed"]
DROP_COLS = ["instant", "dteday", "yr", "n_hours_recorded",
             "season", "is_weekend", "workingday", "t2"]  # STEP7 VIF 실측으로 확정
```

워싱턴과 다른 점: 런던엔 `casual`/`registered`(누수 변수)가 없는 대신, `yr`(Train/Test
각각 단일값이라 무의미)과 `n_hours_recorded`(데이터 품질 메타컬럼)가 새로 제외 대상에
들어간다.

## STEP 1. 데이터 이해 및 탐색 (EDA)

`src/preprocessing.py`로 확인:

- Train(362일)/Test(365일) 모두 **결측치 0건, 완전 중복 0건**
- IQR(1.5×IQR) 기준 이상치는 `t1`/`t2`/`hum`/`wind_speed`/`cnt`에 소수 존재(예: Train
  `wind_speed` 7건, `cnt` 2건)하지만, 폭염·강풍 등 실제 관측값으로 판단해 보정하지 않음
  (워싱턴의 `hum=0` 같은 물리적 모순 사례는 런던 데이터에서 발견되지 않았음)

## STEP 2. 결측치 처리

**해당 없음.** 위 STEP1에서 확인한 대로 결측치가 없어 별도 처리가 필요하지 않았다.
(시간별 원본에는 130시간의 gap이 있었지만, 그 처리는 이미 시간별→일별 변환 단계에서
`n_hours_recorded` 컬럼으로 흡수했다 — `hourly_to_daily_aggregation_rules.md` 참고.)

## STEP 3. 이상치 탐지 및 처리

STEP1에서 발견한 IQR 이상치를 입력 오류가 아닌 실제 관측값으로 판단해 **삭제·보정하지
않고 그대로 사용**했다. 대신 이상치 자체를 설명하는 보조 분석은 STEP과 별도로
`src/anomaly_detection.py`(IsolationForest + counterfactual)로 수행했다 — 결과는
README.md "보조 EDA" 절 참고.

## STEP 4. 날짜 처리 및 파생 변수 생성

시간별 → 일별 변환 시점(`src/hourly_to_daily.py`)에 이미 생성:

| 파생 컬럼 | 방법 |
|---|---|
| `dteday` | `timestamp.dt.date` |
| `yr` | `timestamp.dt.year` |
| `mnth` | `timestamp.dt.month` |
| `weekday` | `(timestamp.dt.dayofweek + 1) % 7` — 워싱턴 `day.csv`와 동일하게 **일요일=0**으로 맞춤(pandas 기본은 월요일=0이라 그대로 두면 인코딩이 어긋남) |
| `workingday` | `(is_holiday==0) & (is_weekend==0)` |
| `n_hours_recorded` | 그 날짜의 원본 시간별 행 수 |

`mnth`는 순환형(sin/cos) 인코딩 대신 원-핫으로 처리했다 — STEP7에서 실측한 VIF가
5.29 이하로 통과해 굳이 순환 인코딩까지 갈 필요가 없었다.

## STEP 5. 데이터 누수(leakage) 변수 제거

런던 원본엔 워싱턴의 `casual`/`registered`(합치면 `cnt`가 되는 변수) 같은 명백한 누수
변수가 없다. 대신 아래 두 컬럼을 "누수는 아니지만 학습에 무의미하거나 부적절한 입력"으로
분류해 제외했다.

- `yr`: Train은 전부 2015, Test는 전부 2016이라 분할 내에서 분산이 0 — 학습에 아무
  정보가 없다.
- `n_hours_recorded`: `cnt`가 부분 합계인지 나타내는 **데이터 품질 지표**이지, 자전거
  수요를 설명하는 변수가 아니다. 모델 입력에서 빼고, 대신 잔차 분석(`predict_2016.py`)
  에서 "이 값이 낮은 날 오차가 큰가"를 검증하는 용도로 썼다 — 실제로 24시간 미만
  기록일의 평균 절대오차(4635.5)가 온전한 날(3104.7)보다 컸다.

## STEP 6. 범주형 변수 인코딩

`pd.get_dummies(..., drop_first=True)`는 워싱턴과 동일하지만, **Train과 Test를 합쳐서
인코딩한 뒤 다시 분리**하는 절차를 추가했다(`build_design_matrix()`).

이유: `weather_code=26`(눈)이 Train(2015)엔 0건, Test(2016)엔 1건뿐이다. 각자
따로 `get_dummies`를 하면 Test에만 `weather_code_26` 컬럼이 생겨 두 데이터의 컬럼
수가 달라지고, 그대로 `predict()`에 넣으면 에러가 난다. 합쳐서 인코딩하면 Train에서
그 컬럼이 전부 0(분산 0)이라 계수가 0으로 추정될 뿐, 구조적으로는 항상 안전하다.

## STEP 7. 다중공선성(multicollinearity) 점검

Train(2015) 설계행렬에 `statsmodels.stats.outliers_influence.variance_inflation_factor`를
3라운드에 걸쳐 적용했다.

| 라운드 | 대상 | 결과 | 조치 |
|---|---|---|---|
| 1차 | `season`+`mnth`, `is_holiday`+`is_weekend`+`workingday` 전체 | 전부 VIF=inf | `season`(월에서 100% 결정 — 크로스탭으로 재확인: 12개월이 각각 정확히 하나의 계절에만 대응) 제거. `workingday`(휴일+주말로 100% 결정) 제거. `is_weekend`(요일 원-핫의 선형결합과 완전히 같음, `is_weekend = 1 - Σ평일더미`) 제거 |
| 2차 | `t1`, `t2` | `t1`=127.2, `t2`=124.9 (상관계수 0.992) | `t2` 제거, `t1`만 유지 |
| 3차(최종) | 잔여 변수 전체 | 최대 VIF=5.29(`t1`), 나머지 대부분 1~5 | 통과, 더 이상 제거 없음 |

워싱턴의 `workingday`(VIF=inf)/`atemp`(VIF≈650) 제거 사례와 원인 구조가 사실상
동일하다 — "합쳐서 만든 파생변수(workingday/season)"와 "서로 거의 같은 정보를 담은
쌍(atemp·temp ↔ t2·t1)"이라는 두 가지 전형적 다중공선성 패턴이 런던에서도 그대로
재현됐다.

## STEP 8. 스케일링/정규화

Ridge/Lasso 튜닝(`tune_linear.py`)에서 `StandardScaler` + 모델을 `Pipeline`으로 묶어
사용했다(워싱턴과 동일). OLS/Random Forest/XGBoost는 스케일에 영향받지 않아 원본
스케일을 그대로 사용했다.

## STEP 9. 학습/테스트 데이터 분할

워싱턴은 시간순 비율 분할(80/20, 585일/146일)을 썼지만, 런던은 **요청에 따라 연도
단위 고정 분할**을 썼다 — Train=2015년(362일), Test=2016년(365일), 2017년 3일은
제외(`split_daily_by_year.py`). "한 번도 보지 않은 해를 예측한다"는 점에서 워싱턴보다
더 엄격한 시간순 분할이다.

하이퍼파라미터 튜닝 시 교차검증(`TimeSeriesSplit`, 5-fold)은 Train(2015) 내부에서만
수행했다. 다만 Train이 362일뿐이라 fold당 약 60일씩만 배정되며, 실제로 Random Forest
튜닝에서는 이 내부 CV 평균 R²가 **-0.1547**로 나올 만큼 불안정했다(README.md 참고) —
표본이 작을 때 시간순 CV 자체의 신뢰도도 함께 낮아진다는 점을 실측으로 확인했다.

## STEP 10~11. 회귀모델 적합 및 평가

OLS(`regression.py`, statsmodels 계수·p-value 확인용), Ridge/Lasso(`tune_linear.py`),
Random Forest(`random_forest.py`, `tune_random_forest.py`), XGBoost(`xgboost_model.py`,
`tune_xgboost.py`) 총 5개 모델 계열, 8개 설정을 Test(2016) 기준으로 비교했다(결과표는
README.md). **최종 채택: Ridge(α=10), Test R²=0.7713.**

## STEP 12. 회귀 가정 진단

`regression.py` 실행 시 statsmodels OLS 요약에서 함께 출력된 값으로 확인했다(별도
진단 스크립트는 워싱턴에도 없어 런던에서도 만들지 않았다 — 요약 출력만으로 충분히
판단 가능했음).

- **독립성 (Durbin-Watson)**: 1.635 — 2(무상관 기준)에 약간 못 미쳐 잔차에 경미한
  양의 자기상관 가능성이 있음. 일별 시계열 데이터 특성상 어느 정도는 예상된 결과.
- **정규성 (Jarque-Bera)**: 통계량 2791.9, p<0.001로 **정규성 기각** — 왜도(skew)
  1.585, 첨도(kurtosis) 16.23으로 꼬리가 매우 두꺼움. 대여량이 폭증하는 소수의 날
  (연휴 성수기, 이상기후 등)이 잔차 분포를 오른쪽으로 길게 늘어뜨린 것으로 해석된다.
- **다중공선성 (Condition Number)**: 1.01e+16으로 매우 큼 — 다만 이는 실제
  다중공선성이 아니라 `weather_code_26.0`(Train에서 완전히 0인 컬럼)이 만든 인위적
  특이성이다. STEP6에서 설명한 대로 예측 안정성에는 문제가 없음을 실측으로 확인했다
  (Test 예측이 정상적으로 산출됨).
- **등분산성 (Breusch-Pagan)**: LM 통계량 38.66(p=0.0525), F 통계량 1.61(p=0.0350).
  LM 검정 기준으론 5% 유의수준을 근소하게 통과(등분산 기각 안 됨)하지만 F 검정
  기준으론 기각된다 — 두 검정이 서로 다른 근사를 쓰기 때문에 경계선에서 갈릴 수
  있으며, 종합하면 "약한 이분산성 의심" 정도로 해석한다. 정규성 위반(위 Jarque-Bera)
  과 같은 원인 — 대여량이 급증하는 소수의 날이 잔차 분산을 키우는 것으로 보인다.
- **결론**: OLS 계수의 유의성 해석(p-value)은 정규성 위반으로 다소 보수적으로 봐야
  하지만, 예측 성능 비교(R²/RMSE/MAE) 자체는 이 가정에 의존하지 않으므로 STEP10~11의
  모델 비교 결론에는 영향이 없다.

## STEP 13. 결과 해석 및 리포팅

- 최종 채택 모델(Ridge)과 특성 중요도(RF/XGBoost 기준)는 README.md에 정리.
- 워싱턴과 달리 **선형 계열이 트리 계열을 앞선 이유**를 "Train 표본 크기(362일 vs
  585일)"로 해석 — STEP9에서 실측한 불안정한 내부 CV(R²=-0.15)가 근거.
- `weather_code`의 일별 대표값(최빈값) 규칙은 런던에 대조용 정답 파일이 없어 검증
  불가능한 근사치라는 캐비앗을 모델 해석에도 유지한다 — 이 변수의 계수/중요도는
  다른 변수보다 신뢰도가 낮게 취급해야 한다.
- 최종 예측·잔차는 `data/processed/london_daily_2016_predictions.csv`, 해석은
  README.md "최종 예측" 절 참고.
