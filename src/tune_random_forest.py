"""
Random Forest 하이퍼파라미터 튜닝 실험
- 목적: "RF를 튜닝하면 선형회귀보다 나아지는가?"를 실측으로 확인
- 방법: 학습 구간(585일) 안에서 TimeSeriesSplit(시간순 교차검증)으로 그리드서치 →
  최적 하이퍼파라미터를 찾은 뒤, 한 번도 보지 않은 테스트 구간(146일)에서 최종 평가
- 비교 대상: 기본 설정 RF, 튜닝된 RF, 강하게 정규화한 RF(비교용), 선형회귀
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

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

    # ---- 1) 기본 설정 RF (지금까지 써온 것, 비교 기준) ----
    baseline = RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
    baseline.fit(X_train, y_train)
    evaluate("RF 기본설정 (n_estimators=300만)", y_test, baseline.predict(X_test))

    # ---- 2) 그리드서치 튜닝 (학습 구간 내부에서만 시간순 교차검증) ----
    param_grid = {
        "n_estimators": [300],
        "max_depth": [3, 6, 10, None],
        "min_samples_leaf": [1, 5, 15],
        "max_features": ["sqrt", None],
    }
    tscv = TimeSeriesSplit(n_splits=5)
    search = GridSearchCV(
        RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        param_grid,
        cv=tscv,
        scoring="r2",
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    print(f"\n[그리드서치] 최적 하이퍼파라미터: {search.best_params_}")
    print(f"[그리드서치] 학습 구간 내 시간순 교차검증 평균 R^2: {search.best_score_:.4f}")

    tuned = search.best_estimator_
    evaluate("RF 튜닝됨 (그리드서치 최적)", y_test, tuned.predict(X_test))
    evaluate("RF 튜닝됨 (train, 참고)", y_train, tuned.predict(X_train))

    # ---- 3) 의도적으로 강하게 정규화한 RF (트리를 아주 얕고 단순하게) ----
    strong = RandomForestRegressor(
        n_estimators=300, max_depth=3, min_samples_leaf=30,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    strong.fit(X_train, y_train)
    evaluate("RF 강한 정규화 (depth=3, leaf>=30)", y_test, strong.predict(X_test))
    evaluate("RF 강한 정규화 (train, 참고)", y_train, strong.predict(X_train))

    # ---- 참고: 선형회귀 (동일 분할) ----
    import statsmodels.api as sm
    X_train_c = sm.add_constant(X_train)
    X_test_c = sm.add_constant(X_test, has_constant="add")
    lr = sm.OLS(y_train, X_train_c).fit()
    print()
    evaluate("선형회귀 (참고, 동일 분할)", y_test, lr.predict(X_test_c))


if __name__ == "__main__":
    run()
