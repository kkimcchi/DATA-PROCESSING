"""
london_daily_2015/2016.csv STEP 1: EDA 재확인 스크립트 (전처리 자체는 불필요함을 확인)
- Train/Test는 이미 hourly_to_daily.py에서 검증된 규칙으로 생성됨(결측 0, 중복 0 확인됨)
- 여기서는 모델링 전 마지막으로 결측/중복/이상치 상태를 다시 확인만 하고, 값은 바꾸지 않는다
  (docs/london_regression_analysis_plan.md STEP1~3 근거 참고 — IQR 이상치는 실제 관측값이라
  임의로 삭제하지 않기로 이미 결정됨)
- 실행: 프로젝트 루트(02-london/)든 src/든 어디서 실행해도 동작
"""
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_PATH = BASE_DIR / "data" / "processed" / "london_daily_2015.csv"
TEST_PATH = BASE_DIR / "data" / "processed" / "london_daily_2016.csv"

NUMERIC_COLS = ["t1", "t2", "hum", "wind_speed", "cnt"]


def check(name: str, df: pd.DataFrame):
    print(f"=== {name}: {df.shape[0]}행 x {df.shape[1]}열 ===")
    print(f"결측치 총합: {df.isna().sum().sum()}")
    print(f"완전 중복 행: {df.duplicated().sum()}")
    for col in NUMERIC_COLS:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = ((df[col] < lower) | (df[col] > upper)).sum()
        print(f"  [{col}] IQR 이상치 {n_out}건 (min={df[col].min():.1f}, max={df[col].max():.1f}) "
              f"-> 실제 관측값으로 판단, 삭제/보정 없음")
    print()


def run():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    check("Train (2015)", train)
    check("Test (2016)", test)
    print("[결론] 결측치/중복 없음, 이상치는 실제 관측값 -> 별도 전처리 불필요. "
          "Train/Test 파일을 그대로 모델링에 사용한다.")


if __name__ == "__main__":
    run()
