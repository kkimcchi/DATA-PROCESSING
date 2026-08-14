"""
day.csv 전처리 스크립트 (STEP 0~9)
- 목적/근거/실행 방법의 상세 설명은 README.md 참조
- 진행 상황은 logs/20260814_day_작업로그.md 참조
- 실행: 프로젝트 루트(day/)든 src/든 어디서 실행해도 동작 (경로가 파일 위치 기준 절대경로)
"""
from pathlib import Path

import numpy as np
import pandas as pd

# ---- STEP 0: 데이터셋 파라미터 식별 ----
BASE_DIR = Path(__file__).resolve().parent.parent  # 프로젝트 루트 (day/)
RAW_PATH = BASE_DIR / "data" / "raw" / "day.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "day_preprocessed.csv"

TARGET = "cnt"
DATE_COL = "dteday"
CATEGORICAL_COLS = ["season", "yr", "mnth", "holiday", "weekday", "workingday", "weathersit"]
NUMERIC_COLS = ["temp", "atemp", "hum", "windspeed"]
DROP_COLS = ["instant", "casual", "registered"]  # 식별자 + 데이터 누수 변수


def load_data(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    return df


def fix_hum_zero(df: pd.DataFrame) -> pd.DataFrame:
    """STEP 3: hum=0(습도 0%)은 물리적으로 비정상 → 결측으로 간주 후 시계열 선형 보간.

    근거: 2011-03-10(weathersit=3, 악천후)에 hum=0.0000으로 기록된 1건이 발견됨.
    악천후인데 습도가 0%인 것은 모순이며, 앞뒤 날짜(0.7754 / 0.6496)는 정상 범위라
    센서 결측이 0으로 잘못 채워진 것으로 판단. 날짜순 정렬된 시계열이므로 선형 보간이
    전체/그룹 평균 대체보다 그날의 실제 기상 맥락에 더 가깝다.
    """
    df = df.copy()
    n_zero = (df["hum"] == 0).sum()
    df.loc[df["hum"] == 0, "hum"] = np.nan
    df["hum"] = df["hum"].interpolate(method="linear")
    return df, n_zero


def run():
    df = load_data()
    df, n_zero = fix_hum_zero(df)

    print(f"[STEP 3] hum=0 -> NaN -> 선형 보간 처리: {n_zero}건")
    fixed_row = df.loc[df[DATE_COL] == "2011-03-10", ["instant", DATE_COL, "hum"]]
    print(fixed_row.to_string(index=False))
    print(f"[검증] 결측치 총합: {df.isnull().sum().sum()}")

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"[저장] {OUTPUT_PATH} ({df.shape[0]}행 x {df.shape[1]}열)")


if __name__ == "__main__":
    run()
