from pathlib import Path
import sys

PROJECT_DIRECTORY = Path(__file__).resolve().parent
SRC_DIRECTORY = PROJECT_DIRECTORY / "src"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from space_weather.pipeline import run_pipeline
from space_weather.visualization import (
    summarize_window,
)


DATA_PATH = PROJECT_DIRECTORY / "models" / "space_weather_data.pkl"
MODEL_PATH = PROJECT_DIRECTORY / "models" / "launch_decision_model.pkl"

COLOR_MAP = {
    "LOW": "#22c55e",
    "MODERATE": "#f59e0b",
    "HIGH": "#f97316",
    "EXTREME": "#dc2626",
}


@st.cache_data(show_spinner=False)
def load_risk_data(_data_path: Path, _model_path: Path) -> tuple[dict, pd.DataFrame]:
    package = joblib.load(_data_path)
    frame = pd.DataFrame(package["risk_features"])
    frame["date"] = pd.to_datetime(frame["date"])
    return package, frame


def rebuild_data() -> None:
    with st.spinner("Cleaning data and retraining the launch-risk model..."):
        run_pipeline(
            PROJECT_DIRECTORY,
            show_dashboard=False,
        )
    st.cache_data.clear()
    st.success("Dashboard data rebuilt successfully.")


def render_risk_chart(window: pd.DataFrame) -> None:
    fig, axis = plt.subplots(figsize=(11, 4))
    colors = window["risk_level"].map(COLOR_MAP).tolist()
    axis.bar(window["date"], window["risk_score"], color=colors, width=0.7)
    axis.axhline(20, color="#57606a", linestyle="--", linewidth=1)
    axis.axhline(60, color="#57606a", linestyle=":", linewidth=1)
    axis.set_title("Daily Launch-Risk Score")
    axis.set_ylabel("Risk score (0-100)")
    axis.set_ylim(0, 105)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    st.pyplot(fig)


def render_event_chart(window: pd.DataFrame) -> None:
    fig, axis = plt.subplots(figsize=(11, 4))
    axis.bar(
        window["date"],
        window["xclass_flare_count"],
        label="X-class",
        color="#dc2626",
        width=0.7,
    )
    axis.bar(
        window["date"],
        window["mclass_flare_count"],
        bottom=window["xclass_flare_count"],
        label="M-class",
        color="#f97316",
        width=0.7,
    )
    axis.bar(
        window["date"],
        window["cclass_flare_count"],
        bottom=window["xclass_flare_count"] + window["mclass_flare_count"],
        label="C-class",
        color="#eab308",
        width=0.7,
    )
    axis.bar(
        window["date"],
        window["storm_count"],
        bottom=(
            window["xclass_flare_count"]
            + window["mclass_flare_count"]
            + window["cclass_flare_count"]
        ),
        label="Storms",
        color="#7c5cd8",
        width=0.7,
    )
    axis.set_title("Solar Events in the Previous 48 Hours")
    axis.set_ylabel("Event count")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=4)
    fig.tight_layout()
    st.pyplot(fig)


def main() -> None:
    st.set_page_config(
        page_title="Space Weather Launch-Risk",
        page_icon="",
        layout="wide",
    )

    st.sidebar.title("Mission Control")
    st.sidebar.caption("Historical space-weather risk analysis")
    if st.sidebar.button("Rebuild data"):
        rebuild_data()

    if not DATA_PATH.is_file() or not MODEL_PATH.is_file():
        rebuild_data()

    package, all_data = load_risk_data(DATA_PATH, MODEL_PATH)

    minimum_date = all_data["date"].min().date()
    maximum_date = all_data["date"].max().date()

    default_start = max(minimum_date, pd.Timestamp("2025-05-01").date())
    default_end = min(maximum_date, pd.Timestamp("2025-07-09").date())

    start_date = st.sidebar.date_input(
        "Start date",
        value=default_start,
        min_value=minimum_date,
        max_value=maximum_date,
    )
    end_date = st.sidebar.date_input(
        "End date",
        value=default_end,
        min_value=minimum_date,
        max_value=maximum_date,
    )

    if start_date > end_date:
        st.sidebar.error("Start date must be on or before end date.")
        st.stop()

    window, summary = summarize_window(
        all_data,
        str(start_date),
        str(end_date),
    )

    st.title("Space Weather Launch-Risk Dashboard")
    st.caption(
        "Educational launch-risk analysis based on historical solar and "
        "geomagnetic activity."
    )

    if window.empty:
        st.warning("No risk data is available in the selected date range.")
        st.stop()

    recommendation = summary["overall_recommendation"]
    recommendation_color = {
        "GO": "green",
        "CAUTION": "orange",
        "DELAY": "orange",
        "NO-GO": "red",
    }[recommendation]

    metric_columns = st.columns(5)
    metric_columns[0].metric("Overall recommendation", recommendation)
    metric_columns[1].metric("Average risk", f"{summary['avg_risk_score']:.1f}")
    metric_columns[2].metric("GO days", summary["go_days"])
    metric_columns[3].metric("Caution days", summary["caution_days"])
    metric_columns[4].metric("NO-GO days", summary["nogo_days"])

    st.markdown(
        f"<h3 style='color:{recommendation_color};'>"
        f"{recommendation} launch window"
        "</h3>",
        unsafe_allow_html=True,
    )

    tab_risk, tab_events, tab_data = st.tabs(
        ["Risk trend", "Solar events", "Daily data"]
    )
    with tab_risk:
        render_risk_chart(window)
    with tab_events:
        render_event_chart(window)
    with tab_data:
        display = window.copy()
        display["date"] = display["date"].dt.strftime("%Y-%m-%d")
        st.dataframe(display, use_container_width=True, hide_index=True)

    st.divider()
    latest = package["current_stats"]
    st.subheader("Latest available conditions")
    current_columns = st.columns(4)
    current_columns[0].metric(
        "Latest date",
        str(pd.Timestamp(latest["latest_date"]).date()),
    )
    current_columns[1].metric(
        "Risk score",
        f"{latest['latest_risk_score']:.1f}/100",
    )
    current_columns[2].metric("Risk level", latest["latest_risk_level"])
    current_columns[3].metric("Recommendation", latest["recommendation"])


if __name__ == "__main__":
    main()
