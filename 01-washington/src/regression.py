"""
day.csv 선형회귀 스크립트 (STEP 6, 10~11 일부)
- 입력: preprocessing.py가 만든 data/processed/day_preprocessed.csv (instant/dteday만 제외한
  전 컬럼에서 누수 변수 casual/registered도 제외, 나머지를 입력값으로 사용)
- 목적/근거/실행 방법의 상세 설명은 README.md 참조
- 실행: 프로젝트 루트(day/)든 src/든 어디서 실행해도 동작 (경로가 파일 위치 기준 절대경로)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent.parent  # 프로젝트 루트 (day/)
INPUT_PATH = BASE_DIR / "data" / "processed" / "day_preprocessed.csv"
TARGET = "cnt"

DROP_COLS = [
    "instant", "dteday", "casual", "registered",  # 번호/날짜 + 데이터 누수 변수
    "workingday",  # weekday + holiday로 완전히 결정되는 파생변수 (VIF=inf, STEP 7에서 제거)
    "atemp",       # temp와 거의 동일한 정보 (VIF≈650, STEP 7에서 제거, temp만 유지)
]
NOMINAL_COLS = ["season", "mnth", "weekday", "weathersit"]  # 원-핫 인코딩 대상 (명목형)
BINARY_NUMERIC_COLS = ["yr", "holiday"]                      # 이미 0/1이라 그대로 사용
CONTINUOUS_COLS = ["temp", "hum", "windspeed"]


def build_design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    X = df.drop(columns=DROP_COLS + [TARGET])
    X = pd.get_dummies(X, columns=NOMINAL_COLS, drop_first=True)
    # get_dummies가 만든 bool 컬럼을 int로 통일 (statsmodels 호환)
    bool_cols = X.select_dtypes(include="bool").columns
    X[bool_cols] = X[bool_cols].astype(int)
    return X


def run():
    df = pd.read_csv(INPUT_PATH)
    y = df[TARGET]
    X = build_design_matrix(df)

    print(f"[입력 변수] {X.shape[1]}개 (원-핫 인코딩 후) — {list(X.columns)}")
    print(f"[표본 수] {X.shape[0]}행\n")

    X_const = sm.add_constant(X.astype(float))
    model = sm.OLS(y, X_const).fit()
    print(model.summary())

    y_pred = model.predict(X_const)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    print(f"\n[전체 데이터 적합 성능] R^2={r2:.4f}  RMSE={rmse:.1f}  MAE={mae:.1f}")


if __name__ == "__main__":
    run()
