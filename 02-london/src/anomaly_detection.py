"""
london_daily.csv 이상치 탐지 스크립트 (IsolationForest 기반 EDA, 회귀 파이프라인과 별도)
- 회귀 Train/Test(2015/2016)와 무관하게 전체 기간(2015-01-04~2017-01-03, 730일)을 대상으로 함
  (비지도 탐지라 데이터 누수 문제가 없음)
- 원본 데이터(data/processed/london_daily.csv)는 읽기 전용으로만 사용, 수정하지 않음
- 실행: 프로젝트 루트(02-london/)든 src/든 어디서 실행해도 동작
"""
from pathlib import Path

import pandas as pd
from sklearn.ensemble import IsolationForest

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = BASE_DIR / "data" / "processed" / "london_daily.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "london_daily_anomalies.csv"

DATE_COL = "dteday"
FEATURES = ["t1", "t2", "hum", "wind_speed", "weather_code", "cnt"]
CONTAMINATION = 0.05
N_ESTIMATORS = 300
RANDOM_STATE = 42


def load_data(path=RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    return df


def detect_anomalies(df: pd.DataFrame):
    X = df[FEATURES].copy()
    model = IsolationForest(
        n_estimators=N_ESTIMATORS, contamination=CONTAMINATION, random_state=RANDOM_STATE
    )
    model.fit(X)
    df = df.copy()
    df["anomaly_score"] = model.decision_function(X)
    df["is_anomaly"] = model.predict(X) == -1
    return df, model, X


def find_top_cause(df: pd.DataFrame, model: IsolationForest, X: pd.DataFrame):
    """counterfactual 민감도 분석: 변수 하나를 중앙값으로 치환했을 때 이상 점수가
    가장 크게 정상화(상승)되는 변수를 그 날의 "가장 직접적인 원인"으로 채택한다.
    (근거는 01-washington/README.md "이상치 탐지" 절 참고 — 동일 방법론)
    """
    median = X.median()
    anomalies = df[df["is_anomaly"]].copy()

    top_causes, top_deltas = [], []
    for idx in anomalies.index:
        base_score = df.loc[idx, "anomaly_score"]
        best_feat, best_delta = None, -float("inf")
        for feat in FEATURES:
            mod = X.loc[[idx]].copy()
            mod[feat] = median[feat]
            new_score = model.decision_function(mod)[0]
            delta = new_score - base_score
            if delta > best_delta:
                best_feat, best_delta = feat, delta
        top_causes.append(best_feat)
        top_deltas.append(best_delta)

    anomalies["top_cause"] = top_causes
    anomalies["top_cause_delta"] = top_deltas
    return anomalies.sort_values("anomaly_score")


def run():
    df = load_data()
    df, model, X = detect_anomalies(df)
    anomalies = find_top_cause(df, model, X)

    print(f"[탐지] 전체 {len(df)}일 중 이상치 {len(anomalies)}일 "
          f"({len(anomalies) / len(df) * 100:.1f}%)\n")
    print(anomalies["top_cause"].value_counts().rename("빈도").to_string())

    cols = [DATE_COL, "cnt", "t1", "t2", "hum", "wind_speed", "weather_code",
            "anomaly_score", "top_cause", "top_cause_delta"]
    anomalies[cols].to_csv(OUTPUT_PATH, index=False)
    print(f"\n[저장] {OUTPUT_PATH} ({len(anomalies)}행)")


if __name__ == "__main__":
    run()
