"""
london_daily.csv를 연도별(2015 / 2016)로 분리해 저장하는 스크립트
- 2017년 데이터(3일치, 연초 잔여분)는 제외
- 입력 파일은 읽기 전용으로만 사용, 수정하지 않음
"""
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DAILY_PATH = BASE_DIR / "data" / "processed" / "london_daily.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

YEARS = [2015, 2016]


def run():
    daily = pd.read_csv(DAILY_PATH)

    print(f"[입력] {DAILY_PATH} 총 {len(daily)}행")
    print(daily["yr"].value_counts().sort_index().to_string())

    for yr in YEARS:
        subset = daily[daily["yr"] == yr].reset_index(drop=True)
        out_path = OUTPUT_DIR / f"london_daily_{yr}.csv"
        subset.to_csv(out_path, index=False)
        print(f"[저장] {out_path} ({subset.shape[0]}행 x {subset.shape[1]}열)")

    excluded = daily[~daily["yr"].isin(YEARS)]
    print(f"[제외] 2017년: {len(excluded)}행 ({excluded['dteday'].tolist()})")


if __name__ == "__main__":
    run()
