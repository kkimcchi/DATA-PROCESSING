"""
런던 일별 자전거 대여 데이터 XGBoost 회귀 스크립트
- 입력 변수는 regression.py(선형회귀), random_forest.py(RF)와 동일하게 맞춰 세 모델을 비교
- 전체 데이터(2015+2016) 적합 성능과, train_test_eval.py와 동일한 2015->2016 평가를 함께 출력
- 실행: 프로젝트 루트(02-london/)든 src/든 어디서 실행해도 동작
"""
import pandas as pd
from xgboost import XGBRegressor

from regression import build_design_matrix, load_train_test
from random_forest import RANDOM_STATE
from train_test_eval import evaluate

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
    train, test = load_train_test()
    X_train, X_test, y_train, y_test = build_design_matrix(train, test)
    X_full = pd.concat([X_train, X_test]).reset_index(drop=True)
    y_full = pd.concat([y_train, y_test]).reset_index(drop=True)

    print(f"[입력 변수] {X_full.shape[1]}개 (선형회귀·RF와 동일)")
    print(f"[표본 수] {X_full.shape[0]}행 (2015+2016 전체)\n")

    # ---- 전체 데이터 적합 (참고용, 과적합 가능성 있음) ----
    full_model = build_model()
    full_model.fit(X_full, y_full)
    full_pred = full_model.predict(X_full)
    print("=== 전체 데이터 적합 (참고, train==test) ===")
    evaluate("XGBoost (전체 적합)", y_full, full_pred)

    importance = (
        pd.Series(full_model.feature_importances_, index=X_full.columns)
        .sort_values(ascending=False)
    )
    print("\n[특성 중요도, 상위 10개]")
    print(importance.head(10).to_string())

    # ---- Train(2015)->Test(2016) 평가 (train_test_eval.py와 동일 분할) ----
    print("\n=== 2015->2016 평가 ===")
    print(f"[분할] 학습 {len(X_train)}행 (2015년) / 테스트 {len(X_test)}행 (2016년)")

    split_model = build_model()
    split_model.fit(X_train, y_train)
    test_pred = split_model.predict(X_test)
    train_pred = split_model.predict(X_train)
    evaluate("XGBoost (2015->2016)", y_test, test_pred)
    evaluate("XGBoost (2015->2015, 참고)", y_train, train_pred)


if __name__ == "__main__":
    run()
