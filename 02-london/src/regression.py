"""
런던 일별 자전거 대여 데이터 선형회귀 스크립트 (STEP 0, 4~7, 10~11)
- Train=2015년(362일) / Test=2016년(365일), 연도 단위 고정 분할
  (docs/london_regression_analysis_plan.md 참고 — 요청에 따라 시간순 비율 분할이 아니라
  연도 경계로 고정)
- STEP 7 다중공선성(VIF)을 Train 기준으로 실측해 DROP_COLS를 확정함 (아래 주석 참고)
- 실행: 프로젝트 루트(02-london/)든 src/든 어디서 실행해도 동작
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_PATH = BASE_DIR / "data" / "processed" / "london_daily_2015.csv"
TEST_PATH = BASE_DIR / "data" / "processed" / "london_daily_2016.csv"

TARGET = "cnt"

# VIF 실측 결과(Train=2015, statsmodels variance_inflation_factor, STEP7):
#   1차: is_holiday/is_weekend/workingday 전부 inf, season/mnth 전부 inf
#   -> season(월에서 100% 결정), workingday(is_holiday+is_weekend로 100% 결정),
#      is_weekend(요일 원-핫과 완전 선형종속) 제거
#   2차: t1=127.2, t2=124.9 (상관계수 0.99) -> t2 제거, t1만 유지 (워싱턴 temp/atemp와 동일 패턴)
#   3차(최종): 잔여 변수 최대 VIF=5.29(t1), 전부 5 이하로 통과
DROP_COLS = [
    "instant", "dteday",   # 식별자/날짜
    "yr",                  # Train=2015/Test=2016 각각 단일값 -> 분산 0
    "n_hours_recorded",    # 데이터 품질 메타컬럼(부분합 여부), 예측 입력 아님
    "season", "is_weekend", "workingday", "t2",  # 다중공선성으로 제거 (위 VIF 실측 근거)
]
NOMINAL_COLS = ["mnth", "weekday", "weather_code"]  # 원-핫 인코딩 대상
BINARY_NUMERIC_COLS = ["is_holiday"]                 # 이미 0/1
CONTINUOUS_COLS = ["t1", "hum", "wind_speed"]


def load_train_test():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    return train, test


def build_design_matrix(train: pd.DataFrame, test: pd.DataFrame):
    """train/test를 합쳐서 원-핫 인코딩한 뒤 다시 분리한다.

    Test(2016)에만 존재하는 weather_code=26(눈, 1건)처럼 Train(2015)에는 없는 범주가
    있으면 따로 get_dummies할 경우 두 데이터의 컬럼 수가 달라져 예측이 깨진다. 합쳐서
    인코딩한 뒤 분리하면 두 쪽 다 항상 동일한 컬럼셋을 갖는다.
    """
    combined = pd.concat([train, test], keys=["train", "test"])
    X = combined.drop(columns=DROP_COLS + [TARGET])
    X = pd.get_dummies(X, columns=NOMINAL_COLS, drop_first=True)
    bool_cols = X.select_dtypes(include="bool").columns
    X[bool_cols] = X[bool_cols].astype(int)
    X = X.astype(float)

    X_train = X.loc["train"].reset_index(drop=True)
    X_test = X.loc["test"].reset_index(drop=True)
    y_train = train[TARGET].reset_index(drop=True)
    y_test = test[TARGET].reset_index(drop=True)
    return X_train, X_test, y_train, y_test


def run():
    train, test = load_train_test()
    X_train, X_test, y_train, y_test = build_design_matrix(train, test)

    print(f"[분할] 학습 {len(X_train)}행 (2015년) / 테스트 {len(X_test)}행 (2016년)")
    print(f"[입력 변수] {X_train.shape[1]}개 (원-핫 인코딩 후) — {list(X_train.columns)}\n")

    X_train_c = sm.add_constant(X_train)
    X_test_c = sm.add_constant(X_test, has_constant="add")
    model = sm.OLS(y_train, X_train_c).fit()
    print(model.summary())

    test_pred = model.predict(X_test_c)
    r2 = r2_score(y_test, test_pred)
    rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    mae = mean_absolute_error(y_test, test_pred)
    print(f"\n[테스트(2016) 성능] R^2={r2:.4f}  RMSE={rmse:.1f}  MAE={mae:.1f}")


if __name__ == "__main__":
    run()
