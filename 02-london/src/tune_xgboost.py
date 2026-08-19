"""
XGBoost 하이퍼파라미터 튜닝 실험 (런던, Train=2015 -> Test=2016, tune_random_forest.py와 동일 방법론)
- Train(2015) 안에서 TimeSeriesSplit으로 그리드서치 -> Test(2016)에서 최종 평가
"""
import statsmodels.api as sm
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor

from regression import build_design_matrix, load_train_test
from random_forest import RANDOM_STATE
from train_test_eval import evaluate


def run():
    train, test = load_train_test()
    X_train, X_test, y_train, y_test = build_design_matrix(train, test)
    print(f"[분할] 학습 {len(X_train)}행 (2015년) / 테스트 {len(X_test)}행 (2016년)\n")

    # ---- 1) 기존에 쓰던 설정 (비교 기준) ----
    baseline = XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    baseline.fit(X_train, y_train)
    evaluate("XGB 기존 설정 (depth=4, lr=0.05)", y_test, baseline.predict(X_test))

    # ---- 2) 그리드서치 튜닝 (Train 내부 시간순 교차검증) ----
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
    print(f"[그리드서치] Train 내 시간순 교차검증 평균 R^2: {search.best_score_:.4f}")

    tuned = search.best_estimator_
    evaluate("XGB 튜닝됨 (그리드서치 최적)", y_test, tuned.predict(X_test))
    evaluate("XGB 튜닝됨 (2015->2015, 참고)", y_train, tuned.predict(X_train))

    # ---- 3) 강한 정규화 버전 (얕고 느리게, 트리 수만 줄임) ----
    strong = XGBRegressor(
        n_estimators=100, max_depth=2, learning_rate=0.03,
        subsample=0.7, colsample_bytree=0.7,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    strong.fit(X_train, y_train)
    evaluate("XGB 강한 정규화 (depth=2, 트리 100개)", y_test, strong.predict(X_test))
    evaluate("XGB 강한 정규화 (2015->2015, 참고)", y_train, strong.predict(X_train))

    # ---- 참고: 선형회귀 ----
    X_train_c = sm.add_constant(X_train)
    X_test_c = sm.add_constant(X_test, has_constant="add")
    lr = sm.OLS(y_train, X_train_c).fit()
    print()
    evaluate("선형회귀 (참고, 동일 분할)", y_test, lr.predict(X_test_c))


if __name__ == "__main__":
    run()
