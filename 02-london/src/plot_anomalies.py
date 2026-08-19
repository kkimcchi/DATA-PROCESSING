"""
london_daily.csv 이상치 탐지 결과 시각화 (PNG 저장)
- data/processed/london_daily_anomalies.csv (anomaly_detection.py 실행 결과)를 읽어
  1) 원인 변수별 빈도 막대그래프
  2) 변수별 2년 시계열 + 원인으로 지목된 날 강조 (소형 멀티플)
  을 figures/ 에 PNG로 저장
- anomaly_detection.py를 먼저 실행해 data/processed/london_daily_anomalies.csv가 있어야 함
- 실행: 프로젝트 루트(02-london/)든 src/든 어디서 실행해도 동작
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = BASE_DIR / "data" / "processed" / "london_daily.csv"
ANOMALY_PATH = BASE_DIR / "data" / "processed" / "london_daily_anomalies.csv"
FIG_DIR = BASE_DIR / "figures"

DATE_COL = "dteday"
CAUSE_ORDER = ["weather_code", "wind_speed", "t1", "hum", "cnt", "t2"]
CAUSE_LABEL = {
    "weather_code": "날씨코드", "wind_speed": "풍속", "t1": "기온",
    "hum": "습도", "cnt": "대여수", "t2": "체감기온",
}
CAUSE_COLOR = {
    "weather_code": "#2a78d6", "wind_speed": "#eb6834", "t1": "#1baf7a",
    "hum": "#eda100", "cnt": "#e87ba4", "t2": "#8a6fd6",
}
CONTEXT_COLOR = "#9aa1a8"
WEATHER_LABELS = {1: "맑음", 2: "구름조금", 3: "구름많음", 4: "흐림", 7: "비", 10: "뇌우", 26: "눈"}

plt.rcParams["font.family"] = ["Malgun Gothic", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


def load_data():
    daily = pd.read_csv(RAW_PATH)
    daily[DATE_COL] = pd.to_datetime(daily[DATE_COL])
    daily = daily.sort_values(DATE_COL).reset_index(drop=True)

    anomalies = pd.read_csv(ANOMALY_PATH)
    anomalies[DATE_COL] = pd.to_datetime(anomalies[DATE_COL])
    return daily, anomalies


def plot_cause_frequency(anomalies: pd.DataFrame, out_path: Path):
    counts = anomalies["top_cause"].value_counts().reindex(CAUSE_ORDER).fillna(0).astype(int)
    counts = counts.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    colors = [CAUSE_COLOR[k] for k in counts.index]
    bars = ax.barh([CAUSE_LABEL[k] for k in counts.index], counts.values, color=colors, height=0.6)

    total = len(anomalies)
    for bar, val in zip(bars, counts.values):
        pct = val / total * 100
        ax.text(bar.get_width() + max(counts.values) * 0.015, bar.get_y() + bar.get_height() / 2,
                 f"{val}일 ({pct:.1f}%)", va="center", fontsize=10)

    ax.set_xlabel("이상치로 지목된 일수")
    ax.set_title(f"원인 변수별 빈도 (이상치 {total}일 중, IsolationForest + counterfactual 분석)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, max(counts.values) * 1.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_variable_timeseries(daily: pd.DataFrame, anomalies: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(len(CAUSE_ORDER), 1, figsize=(10, 15), sharex=True)

    for ax, key in zip(axes, CAUSE_ORDER):
        ax.plot(daily[DATE_COL], daily[key], color=CONTEXT_COLOR, linewidth=1, alpha=0.8)

        hit = anomalies[anomalies["top_cause"] == key]
        ax.scatter(hit[DATE_COL], hit[key], color=CAUSE_COLOR[key], s=45,
                   zorder=5, edgecolors="white", linewidths=1)

        n = len(hit)
        ax.set_ylabel(CAUSE_LABEL[key], fontsize=10)
        ax.set_title(f"{CAUSE_LABEL[key]} — {n}일 원인 지목", loc="left", fontsize=10,
                     color=CAUSE_COLOR[key], fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        if key == "weather_code":
            ticks = sorted(daily["weather_code"].unique())
            ax.set_yticks(ticks)
            ax.set_yticklabels([WEATHER_LABELS.get(int(t), str(t)) for t in ticks])

    axes[-1].set_xlabel("날짜")
    fig.suptitle("변수별 2년 시계열 — 이상치 원인으로 지목된 날 강조", fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run():
    daily, anomalies = load_data()

    freq_path = FIG_DIR / "anomaly_cause_frequency.png"
    ts_path = FIG_DIR / "anomaly_variable_timeseries.png"

    plot_cause_frequency(anomalies, freq_path)
    plot_variable_timeseries(daily, anomalies, ts_path)

    print(f"[저장] {freq_path}")
    print(f"[저장] {ts_path}")


if __name__ == "__main__":
    run()
