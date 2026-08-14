"""
XGBoost 하이퍼파라미터 튜닝 실험 (tune_random_forest.py와 동일한 방법론)
- 학습 구간(585일) 안에서 TimeSeriesSplit으로 그리드서치 -> 테스트 구간(146일)에서 최종 평가
- RF 튜닝 실험과 같은 결론(구조적 외삽 한계)이 XGBoost에도 적용되는지 확인
"""
import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor

from regression import INPUT_PATH, TARGET, build_design_matrix
from random_forest import RANDOM_STATE
from train_test_eval import chronological_split, evaluate


def run():
    df = pd.read_csv(INPUT_PATH)
    y = df[TARGET]
    X = build_design_matrix(df).astype(float)
    X_train, X_test, y_train, y_test, train_dates, test_dates = chronological_split(df, X, y)
    print(f"[분할] 학습 {len(X_train)}행 ({train_dates[0]} ~ {train_dates[1]}) / "
          f"테스트 {len(X_test)}행 ({test_dates[0]} ~ {test_dates[1]})\n")

    # ---- 1) 기존에 쓰던 설정 (비교 기준) ----
    baseline = XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    baseline.fit(X_train, y_train)
    evaluate("XGB 기존 설정 (depth=4, lr=0.05)", y_test, baseline.predict(X_test))

    # ---- 2) 그리드서치 튜닝 (학습 구간 내부 시간순 교차검증) ----
    param_grid = {
        "n_estimators": [300],
        "max_depth": [2, 3, 4, 6],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }
    tscv = TimeSeriesSplit(n_splits=5)
    search = GridSearchCV(
        XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        param_grid,
        cv=tscv,
        scoring="r2",
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    print(f"\n[그리드서치] 최적 하이퍼파라미터: {search.best_params_}")
    print(f"[그리드서치] 학습 구간 내 시간순 교차검증 평균 R^2: {search.best_score_:.4f}")

    tuned = search.best_estimator_
    evaluate("XGB 튜닝됨 (그리드서치 최적)", y_test, tuned.predict(X_test))
    evaluate("XGB 튜닝됨 (train, 참고)", y_train, tuned.predict(X_train))

    # ---- 3) 강한 정규화 버전 (얕고 느리게, 조기종료 없이 트리 수만 줄임) ----
    strong = XGBRegressor(
        n_estimators=100, max_depth=2, learning_rate=0.03,
        subsample=0.7, colsample_bytree=0.7,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    strong.fit(X_train, y_train)
    evaluate("XGB 강한 정규화 (depth=2, 트리 100개)", y_test, strong.predict(X_test))
    evaluate("XGB 강한 정규화 (train, 참고)", y_train, strong.predict(X_train))

    # ---- 참고: 선형회귀 ----
    import statsmodels.api as sm
    X_train_c = sm.add_constant(X_train)
    X_test_c = sm.add_constant(X_test, has_constant="add")
    lr = sm.OLS(y_train, X_train_c).fit()
    print()
    evaluate("선형회귀 (참고, 동일 분할)", y_test, lr.predict(X_test_c))


if __name__ == "__main__":
    run()
