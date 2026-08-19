"""
최종 채택 모델(Ridge, alpha=10)로 2016년 cnt를 예측하고 잔차를 분석한다.
- 채택 근거: tune_linear.py 실행 결과 Test(2016) R^2=0.7713으로 8개 모델·설정 중 최고
  (docs/london_regression_analysis_plan.md 7절, README.md 결과표 참고)
- n_hours_recorded<24인 16일(부분 합계로 집계된 날)에 오차가 몰리는지 함께 확인한다
  (계획안 4-1절에서 미리 제기한 검증 항목)
- 실행: 프로젝트 루트(02-london/)든 src/든 어디서 실행해도 동작
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from regression import build_design_matrix, load_train_test

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "london_daily_2016_predictions.csv"

BEST_ALPHA = 10


def run():
    train, test = load_train_test()
    X_train, X_test, y_train, y_test = build_design_matrix(train, test)

    model = make_pipeline(StandardScaler(), Ridge(alpha=BEST_ALPHA))
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    result = test[["dteday", "n_hours_recorded"]].copy()
    result["cnt_actual"] = y_test.values
    result["cnt_predicted"] = pred
    result["residual"] = result["cnt_actual"] - result["cnt_predicted"]
    result["abs_error"] = result["residual"].abs()

    r2 = 1 - np.sum(result["residual"] ** 2) / np.sum((result["cnt_actual"] - result["cnt_actual"].mean()) ** 2)
    rmse = np.sqrt((result["residual"] ** 2).mean())
    mae = result["abs_error"].mean()
    print(f"[최종 모델: Ridge alpha={BEST_ALPHA}] 2016년 예측 성능 R^2={r2:.4f}  RMSE={rmse:.1f}  MAE={mae:.1f}\n")

    incomplete = result[result["n_hours_recorded"] < 24]
    complete = result[result["n_hours_recorded"] == 24]
    print(f"[잔차 비교] 24시간 온전한 날({len(complete)}일) 평균 절대오차: {complete['abs_error'].mean():.1f}")
    print(f"[잔차 비교] 24시간 미만인 날({len(incomplete)}일) 평균 절대오차: {incomplete['abs_error'].mean():.1f}")

    print("\n[오차 상위 10일]")
    print(result.sort_values("abs_error", ascending=False).head(10)
          [["dteday", "n_hours_recorded", "cnt_actual", "cnt_predicted", "residual"]]
          .to_string(index=False))

    result.to_csv(OUTPUT_PATH, index=False)
    print(f"\n[저장] {OUTPUT_PATH} ({len(result)}행)")


if __name__ == "__main__":
    run()
