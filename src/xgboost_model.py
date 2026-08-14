"""
day.csv XGBoost 회귀 스크립트
- 입력 변수는 regression.py(선형회귀), random_forest.py(RF)와 동일하게 맞춰 세 모델을 비교
- 전체 데이터 적합 성능과, train_test_eval.py와 동일한 시간순 분할 기준 성능을 함께 출력
- 실행: 프로젝트 루트(day/)든 src/든 어디서 실행해도 동작
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from regression import INPUT_PATH, TARGET, build_design_matrix
from random_forest import RANDOM_STATE
from train_test_eval import chronological_split, evaluate

N_ESTIMATORS = 300


def build_model():
    return XGBRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=4,
        learning_rate=0.05,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def run():
    df = pd.read_csv(INPUT_PATH)
    y = df[TARGET]
    X = build_design_matrix(df).astype(float)

    print(f"[입력 변수] {X.shape[1]}개 (선형회귀·RF와 동일)")
    print(f"[표본 수] {X.shape[0]}행\n")

    # ---- 전체 데이터 적합 (참고용, 과적합 가능성 있음) ----
    full_model = build_model()
    full_model.fit(X, y)
    full_pred = full_model.predict(X)
    print("=== 전체 데이터 적합 (참고, train==test) ===")
    evaluate("XGBoost (전체 적합)", y, full_pred)

    importance = (
        pd.Series(full_model.feature_importances_, index=X.columns)
        .sort_values(ascending=False)
    )
    print("\n[특성 중요도, 상위 10개]")
    print(importance.head(10).to_string())

    # ---- 시간순 train/test 평가 (train_test_eval.py와 동일 분할) ----
    X_train, X_test, y_train, y_test, train_dates, test_dates = chronological_split(df, X, y)
    print(f"\n=== 시간순 train/test 평가 ===")
    print(f"[분할] 학습 {len(X_train)}행 ({train_dates[0]} ~ {train_dates[1]}) / "
          f"테스트 {len(X_test)}행 ({test_dates[0]} ~ {test_dates[1]})")

    split_model = build_model()
    split_model.fit(X_train, y_train)
    test_pred = split_model.predict(X_test)
    train_pred = split_model.predict(X_train)
    evaluate("XGBoost (train->test)", y_test, test_pred)
    evaluate("XGBoost (train->train, 참고)", y_train, train_pred)


if __name__ == "__main__":
    run()
