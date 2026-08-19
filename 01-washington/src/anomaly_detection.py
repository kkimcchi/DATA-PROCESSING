"""
day.csv 이상치 탐지 스크립트 (IsolationForest 기반 EDA, 회귀 파이프라인과 별도)
- 목적/근거/실행 방법의 상세 설명은 README.md "이상치 탐지" 절 참조
- 실행: 프로젝트 루트(day/)든 src/든 어디서 실행해도 동작 (경로가 파일 위치 기준 절대경로)
- 원본 데이터(data/raw/day.csv)는 읽기 전용으로만 사용, 수정하지 않음
"""
from pathlib import Path

import pandas as pd
from sklearn.ensemble import IsolationForest

# ---- STEP 0: 파라미터 ----
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = BASE_DIR / "data" / "raw" / "day.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "day_anomalies.csv"

DATE_COL = "dteday"
FEATURES = ["temp", "atemp", "hum", "windspeed", "weathersit", "cnt"]
CONTAMINATION = 0.05  # 전체의 약 5%를 이상치로 판정
N_ESTIMATORS = 300
RANDOM_STATE = 42


def load_data(path=RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    return df


def detect_anomalies(df: pd.DataFrame):
    """IsolationForest로 이상치를 탐지하고 anomaly_score, is_anomaly 컬럼을 추가."""
    X = df[FEATURES].copy()
    model = IsolationForest(
        n_estimators=N_ESTIMATORS, contamination=CONTAMINATION, random_state=RANDOM_STATE
    )
    model.fit(X)
    df = df.copy()
    df["anomaly_score"] = model.decision_function(X)  # 낮을수록 더 이상함
    df["is_anomaly"] = model.predict(X) == -1
    return df, model, X


def find_top_cause(df: pd.DataFrame, model: IsolationForest, X: pd.DataFrame):
    """이상치로 판정된 각 날짜에 대해 counterfactual 민감도 분석으로 최직접 원인 변수를 찾는다.

    근거: 특성별 z-score만으로는 "모델이 실제로 그 변수 때문에 이상치라고 판단했는지"를
    확정할 수 없다. 대신 변수 하나를 전체 데이터의 중앙값으로 치환한 뒤 같은 모델로
    이상 점수를 다시 계산해, 점수가 가장 크게 정상화(상승)되는 변수를 그 날의
    "가장 직접적인 원인"으로 채택한다. 이는 z-score 기반 설명과 달리 모델 자체의
    판단에 대한 인과적 기여도를 역산한 값이다.
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

    cols = [DATE_COL, "cnt", "temp", "atemp", "hum", "windspeed", "weathersit",
            "anomaly_score", "top_cause", "top_cause_delta"]
    anomalies[cols].to_csv(OUTPUT_PATH, index=False)
    print(f"\n[저장] {OUTPUT_PATH} ({len(anomalies)}행)")


if __name__ == "__main__":
    run()
