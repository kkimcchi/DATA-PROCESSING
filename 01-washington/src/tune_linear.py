"""
정규화 선형회귀(Ridge, Lasso) 하이퍼파라미터 튜닝 실험
- OLS 선형회귀는 튜닝할 하이퍼파라미터가 없으므로(정규방정식으로 유일하게 결정),
  "선형 계열도 튜닝하면 어떻게 되는가"를 확인하기 위해 alpha(정규화 강도)를 갖는
  Ridge/Lasso를 RF·XGBoost와 동일한 방법론(TimeSeriesSplit 그리드서치)으로 튜닝
- Ridge/Lasso는 변수 스케일에 민감하므로 StandardScaler와 파이프라인으로 묶음
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from regression import INPUT_PATH, TARGET, build_design_matrix
from train_test_eval import chronological_split, evaluate

ALPHAS = np.logspace(-3, 3, 25)  # 0.001 ~ 1000


def tune_and_evaluate(name, estimator_cls, X_train, y_train, X_test, y_test):
    pipe = make_pipeline(StandardScaler(), estimator_cls())
    param_name = f"{estimator_cls.__name__.lower()}__alpha"
    tscv = TimeSeriesSplit(n_splits=5)
    search = GridSearchCV(pipe, {param_name: ALPHAS}, cv=tscv, scoring="r2", n_jobs=-1)
    search.fit(X_train, y_train)

    print(f"[{name}] 최적 alpha={search.best_params_[param_name]:.4g}  "
          f"내부 CV 평균 R^2={search.best_score_:.4f}")
    best = search.best_estimator_
    evaluate(f"{name} 튜닝됨 (test)", y_test, best.predict(X_test))
    evaluate(f"{name} 튜닝됨 (train, 참고)", y_train, best.predict(X_train))
    return best


def run():
    df = pd.read_csv(INPUT_PATH)
    y = df[TARGET]
    X = build_design_matrix(df).astype(float)
    X_train, X_test, y_train, y_test, train_dates, test_dates = chronological_split(df, X, y)
    print(f"[분할] 학습 {len(X_train)}행 ({train_dates[0]} ~ {train_dates[1]}) / "
          f"테스트 {len(X_test)}행 ({test_dates[0]} ~ {test_dates[1]})\n")

    tune_and_evaluate("Ridge", Ridge, X_train, y_train, X_test, y_test)
    print()
    tune_and_evaluate("Lasso", Lasso, X_train, y_train, X_test, y_test)

    # ---- 참고: 일반 OLS (튜닝 대상 없음, 비교 기준) ----
    import statsmodels.api as sm
    X_train_c = sm.add_constant(X_train)
    X_test_c = sm.add_constant(X_test, has_constant="add")
    lr = sm.OLS(y_train, X_train_c).fit()
    print()
    evaluate("OLS 선형회귀 (참고, 튜닝 없음)", y_test, lr.predict(X_test_c))


if __name__ == "__main__":
    run()
