"""
런던 일별 자전거 대여 데이터 랜덤포레스트 회귀 스크립트
- 입력 변수는 regression.py(선형회귀)와 동일하게 맞춰 두 모델을 직접 비교할 수 있게 한다
- 여기서는 2015+2016 전체(train+test 합본)에 그대로 적합한 "전체 데이터 적합" 결과를 보여준다
  (train/test 분리 없이 학습·평가를 같은 데이터로 하면 성능이 얼마나 부풀려지는지 보여주기 위한
  참고용 — 실제 일반화 성능은 train_test_eval.py에서 2015->2016으로 따로 평가한다)
- 실행: 프로젝트 루트(02-london/)든 src/든 어디서 실행해도 동작
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from regression import TARGET, build_design_matrix, load_train_test

N_ESTIMATORS = 300
RANDOM_STATE = 42


def run():
    train, test = load_train_test()
    X_train, X_test, y_train, y_test = build_design_matrix(train, test)
    X_full = pd.concat([X_train, X_test]).reset_index(drop=True)
    y_full = pd.concat([y_train, y_test]).reset_index(drop=True)

    print(f"[입력 변수] {X_full.shape[1]}개 (선형회귀와 동일) — {list(X_full.columns)}")
    print(f"[표본 수] {X_full.shape[0]}행 (2015+2016 전체)\n")

    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1
    )
    model.fit(X_full, y_full)
    y_pred = model.predict(X_full)

    r2 = r2_score(y_full, y_pred)
    rmse = np.sqrt(mean_squared_error(y_full, y_pred))
    mae = mean_absolute_error(y_full, y_pred)
    print(f"[전체 데이터 적합 성능 (참고, 과적합 가능성 있음)] R^2={r2:.4f}  RMSE={rmse:.1f}  MAE={mae:.1f}\n")

    importance = (
        pd.Series(model.feature_importances_, index=X_full.columns)
        .sort_values(ascending=False)
    )
    print("[특성 중요도 (feature_importances_), 상위 10개]")
    print(importance.head(10).to_string())


if __name__ == "__main__":
    run()
