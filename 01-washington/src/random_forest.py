"""
day.csv 랜덤포레스트 회귀 스크립트
- 입력 변수는 regression.py(선형회귀)와 동일하게 맞춰 두 모델을 직접 비교할 수 있게 한다
  (build_design_matrix, DROP_COLS 등을 그대로 재사용)
- 전체 데이터에 그대로 적합한 결과이며, 선형회귀와 동일한 조건(train/test 분할 전)으로 비교
- 실행: 프로젝트 루트(day/)든 src/든 어디서 실행해도 동작
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from regression import INPUT_PATH, TARGET, build_design_matrix

N_ESTIMATORS = 300
RANDOM_STATE = 42


def run():
    df = pd.read_csv(INPUT_PATH)
    y = df[TARGET]
    X = build_design_matrix(df).astype(float)

    print(f"[입력 변수] {X.shape[1]}개 (선형회귀와 동일) — {list(X.columns)}")
    print(f"[표본 수] {X.shape[0]}행\n")

    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1
    )
    model.fit(X, y)
    y_pred = model.predict(X)

    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    print(f"[전체 데이터 적합 성능] R^2={r2:.4f}  RMSE={rmse:.1f}  MAE={mae:.1f}\n")

    importance = (
        pd.Series(model.feature_importances_, index=X.columns)
        .sort_values(ascending=False)
    )
    print("[특성 중요도 (feature_importances_), 상위 10개]")
    print(importance.head(10).to_string())


if __name__ == "__main__":
    run()
