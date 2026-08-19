"""
london_merged.csv (시간별) -> 일별 변환 스크립트
- 규칙 근거: docs/hourly_to_daily_aggregation_rules.md, docs/london_hourly_to_daily_conversion_plan.md
- 원본 데이터(data/london_merged.csv)는 읽기 전용으로만 사용, 수정하지 않음
- 실행: 프로젝트 루트(02-london/)든 src/든 어디서 실행해도 동작 (경로가 파일 위치 기준 절대경로)
"""
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = BASE_DIR / "data" / "london_merged.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "london_daily.csv"

DATE_COL = "dteday"
MEAN_COLS = ["t1", "t2", "hum", "wind_speed"]
SUM_COLS = ["cnt"]
CARRY_OVER_COLS = ["is_holiday", "is_weekend", "season"]
MODE_COL = "weather_code"

OUTPUT_COL_ORDER = [
    "instant", "dteday", "yr", "mnth", "weekday",
    "is_holiday", "is_weekend", "workingday", "season",
    "weather_code", "t1", "t2", "hum", "wind_speed", "cnt",
    "n_hours_recorded",
]


def load_data(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[DATE_COL] = df["timestamp"].dt.date
    df["yr"] = df["timestamp"].dt.year
    df["mnth"] = df["timestamp"].dt.month
    # pandas dt.dayofweek는 월=0~일=6이지만, 워싱턴 day.csv는 일=0~토=6으로 인코딩되어
    # 있음(역산 확인: 2011-01-01 토요일 -> weekday=6). 같은 인코딩으로 맞추기 위해 +1 shift.
    df["weekday"] = (df["timestamp"].dt.dayofweek + 1) % 7
    df["workingday"] = ((df["is_holiday"] == 0) & (df["is_weekend"] == 0)).astype(int)
    return df


def check_carry_over(df: pd.DataFrame) -> dict:
    """carry-over 규칙 적용 전, 하루 안에서 값이 고정인지 컬럼별로 검증.

    계획안 5절에 따라 값이 갈리는 날이 있으면 그 날짜를 경고 목록으로 남긴다.
    (사전 점검 결과 730일 전부 통과했지만, 실행 스크립트에서도 재확인한다.)
    """
    warnings = {}
    for col in CARRY_OVER_COLS:
        nunique_per_day = df.groupby(DATE_COL)[col].nunique()
        bad_days = nunique_per_day[nunique_per_day > 1].index.tolist()
        if bad_days:
            warnings[col] = bad_days
    return warnings


def mode_weather_code(s: pd.Series):
    """최빈값, 동률이면 숫자가 큰(더 나쁜) 코드 우선.

    검증 캐비앗: 런던엔 대조용 일별 정답 파일이 없어 이 규칙의 일치율은
    측정 불가 (docs/hourly_to_daily_aggregation_rules.md 참조).
    """
    counts = s.value_counts()
    top_count = counts.max()
    tied_codes = counts[counts == top_count].index
    return max(tied_codes)


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    agg_spec = {
        "n_hours_recorded": ("timestamp", "count"),
        "yr": ("yr", "first"),
        "mnth": ("mnth", "first"),
        "weekday": ("weekday", "first"),
        "workingday": ("workingday", "first"),
        "is_holiday": ("is_holiday", "first"),
        "is_weekend": ("is_weekend", "first"),
        "season": ("season", "first"),
        "t1": ("t1", "mean"),
        "t2": ("t2", "mean"),
        "hum": ("hum", "mean"),
        "wind_speed": ("wind_speed", "mean"),
        "cnt": ("cnt", "sum"),
        "weather_code": ("weather_code", mode_weather_code),
    }
    daily = df.groupby(DATE_COL).agg(**agg_spec).reset_index()
    daily = daily.sort_values(DATE_COL).reset_index(drop=True)
    daily.insert(0, "instant", range(1, len(daily) + 1))
    return daily[OUTPUT_COL_ORDER]


def run():
    df = load_data()
    df = add_derived_columns(df)

    warnings = check_carry_over(df)
    if warnings:
        for col, bad_days in warnings.items():
            print(f"[경고] '{col}' 값이 하루 안에서 갈리는 날짜 {len(bad_days)}건: {bad_days}")
    else:
        print("[검증] is_holiday / is_weekend / season 모두 730일 전부 하루 내내 값 고정 확인 (carry-over 안전)")

    daily = aggregate_daily(df)

    incomplete_days = daily[daily["n_hours_recorded"] < 24]

    print(f"[변환 결과] 총 행 수(일 수): {len(daily)}")
    print(f"[변환 결과] 24시간이 온전하지 않은 날: {len(incomplete_days)}일")
    print(incomplete_days[["dteday", "n_hours_recorded"]].to_string(index=False))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(OUTPUT_PATH, index=False)
    print(f"[저장] {OUTPUT_PATH} ({daily.shape[0]}행 x {daily.shape[1]}열)")


if __name__ == "__main__":
    run()
