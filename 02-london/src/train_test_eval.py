"""
런던 일별 데이터 STEP 9: 선형회귀 vs Random Forest, Train(2015)->Test(2016) 평가
- 두 모델 모두 "2015년으로 학습, 2016년으로 평가"로 통일해
  random_forest.py의 전체 데이터 적합(과적합 우려) 결과와 공정하게 비교
- 실행: 프로젝트 루트(02-london/)든 src/든 어디서 실행해도 동작
"""
import numpy as np
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from regression import build_design_matrix, load_train_test
from random_forest import N_ESTIMATORS, RANDOM_STATE


def evaluate(name, y_test, y_pred):
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    print(f"[{name}] R^2={r2:.4f}  RMSE={rmse:.1f}  MAE={mae:.1f}")
    return r2, rmse, mae


def run():
    train, test = load_train_test()
    X_train, X_test, y_train, y_test = build_design_matrix(train, test)
    print(f"[분할] 학습 {len(X_train)}행 (2015년) / 테스트 {len(X_test)}행 (2016년)\n")

    # 선형회귀 (statsmodels로 학습, sklearn 지표 함수로 평가)
    X_train_c = sm.add_constant(X_train)
    X_test_c = sm.add_constant(X_test, has_constant="add")
    lr_model = sm.OLS(y_train, X_train_c).fit()
    lr_pred = lr_model.predict(X_test_c)
    evaluate("선형회귀 (2015->2016)", y_test, lr_pred)

    # Random Forest
    rf_model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    evaluate("Random Forest (2015->2016)", y_test, rf_pred)

    # 참고용: 학습 데이터 자체 성능(과적합 정도 확인용)
    print()
    lr_train_pred = lr_model.predict(X_train_c)
    evaluate("선형회귀 (2015->2015, 참고)", y_train, lr_train_pred)
    rf_train_pred = rf_model.predict(X_train)
    evaluate("Random Forest (2015->2015, 참고)", y_train, rf_train_pred)


if __name__ == "__main__":
    run()
