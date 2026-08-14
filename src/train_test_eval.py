"""
day.csv STEP 9: train/test 분할 + 선형회귀 vs Random Forest 재평가
- 시계열 데이터이므로 무작위 분할이 아닌 시간순 분할 사용 (README STEP 9 근거 참조):
  앞 80%(시간상 과거) 학습 / 뒤 20%(시간상 최근) 테스트
- 두 모델 모두 "학습 데이터에서 학습, 테스트 데이터에서 평가"로 통일해
  이전의 전체 데이터 적합(overfitting 우려가 있던) 결과와 공정하게 비교
- 실행: 프로젝트 루트(day/)든 src/든 어디서 실행해도 동작
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from regression import INPUT_PATH, TARGET, build_design_matrix
from random_forest import N_ESTIMATORS, RANDOM_STATE

TEST_SIZE = 0.2


def chronological_split(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series):
    n_test = int(len(df) * TEST_SIZE)
    n_train = len(df) - n_test
    X_train, X_test = X.iloc[:n_train], X.iloc[n_train:]
    y_train, y_test = y.iloc[:n_train], y.iloc[n_train:]
    train_dates = (df["dteday"].iloc[0], df["dteday"].iloc[n_train - 1])
    test_dates = (df["dteday"].iloc[n_train], df["dteday"].iloc[-1])
    return X_train, X_test, y_train, y_test, train_dates, test_dates


def evaluate(name, y_test, y_pred):
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    print(f"[{name}] R^2={r2:.4f}  RMSE={rmse:.1f}  MAE={mae:.1f}")
    return r2, rmse, mae


def run():
    df = pd.read_csv(INPUT_PATH)
    y = df[TARGET]
    X = build_design_matrix(df).astype(float)

    X_train, X_test, y_train, y_test, train_dates, test_dates = chronological_split(df, X, y)
    print(f"[분할] 학습 {len(X_train)}행 ({train_dates[0]} ~ {train_dates[1]}) / "
          f"테스트 {len(X_test)}행 ({test_dates[0]} ~ {test_dates[1]})\n")

    # 선형회귀 (statsmodels로 학습, sklearn 지표 함수로 평가)
    X_train_const = sm.add_constant(X_train)
    X_test_const = sm.add_constant(X_test, has_constant="add")
    lr_model = sm.OLS(y_train, X_train_const).fit()
    lr_pred = lr_model.predict(X_test_const)
    evaluate("선형회귀 (train->test)", y_test, lr_pred)

    # Random Forest
    rf_model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    evaluate("Random Forest (train->test)", y_test, rf_pred)

    # 참고용: 학습 데이터 자체 성능(과적합 정도 확인용)
    print()
    lr_train_pred = lr_model.predict(X_train_const)
    evaluate("선형회귀 (train->train, 참고)", y_train, lr_train_pred)
    rf_train_pred = rf_model.predict(X_train)
    evaluate("Random Forest (train->train, 참고)", y_train, rf_train_pred)


if __name__ == "__main__":
    run()
