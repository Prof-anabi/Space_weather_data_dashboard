import matplotlib.pyplot as plt
import pandas as pd


RECOMMENDATION_MAP = {
    "LOW": "GO",
    "MODERATE": "CAUTION",
    "HIGH": "DELAY",
    "EXTREME": "NO-GO",
}

COLOR_MAP = {
    "LOW": "#22c55e",
    "MODERATE": "#f59e0b",
    "HIGH": "#f97316",
    "EXTREME": "#dc2626",
}


def summarize_window(
    risk_features: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> tuple[pd.DataFrame, dict]:
    window = risk_features[
        (risk_features["date"] >= pd.Timestamp(start_date))
        & (risk_features["date"] <= pd.Timestamp(end_date))
    ].reset_index(drop=True)

    if len(window):
        dominant_level = window["risk_level"].mode()[0]
        max_index = window["risk_score"].idxmax()
        max_date = str(window.loc[max_index, "date"])[:10]
    else:
        dominant_level = "LOW"
        max_date = "N/A"

    summary = {
        "total_days": len(window),
        "avg_risk_score": round(window["risk_score"].mean(), 2)
        if len(window)
        else 0,
        "go_days": int((window["risk_level"] == "LOW").sum()),
        "caution_days": int((window["risk_level"] == "MODERATE").sum()),
        "delay_days": int((window["risk_level"] == "HIGH").sum()),
        "nogo_days": int((window["risk_level"] == "EXTREME").sum()),
        "overall_recommendation": RECOMMENDATION_MAP[dominant_level],
        "max_risk_score": float(window["risk_score"].max())
        if len(window)
        else 0.0,
        "max_risk_date": max_date,
    }
    return window, summary


def print_window_summary(
    summary: dict,
    start_date: str,
    end_date: str,
) -> None:
    print(f"=== Go/No-Go Analysis: {start_date} to {end_date} ===")
    print(f"Overall Recommendation : {summary['overall_recommendation']}")
    print(f"Days analysed          : {summary['total_days']}")
    print(f"Avg risk score         : {summary['avg_risk_score']:.2f} / 100")
    print(f"GO days                : {summary['go_days']}")
    print(f"CAUTION days           : {summary['caution_days']}")
    print(f"DELAY days             : {summary['delay_days']}")
    print(f"NO-GO days             : {summary['nogo_days']}")
    print(
        "Highest risk date      : "
        f"{summary['max_risk_date']}  "
        f"(score {summary['max_risk_score']:.1f})"
    )


def plot_dashboard(
    window: pd.DataFrame,
    start_date: str,
    end_date: str,
    show: bool = True,
) -> plt.Figure | None:
    if len(window) == 0:
        print("No data in selected window.")
        return None

    bar_colors = window["risk_level"].map(COLOR_MAP).tolist()
    dates = window["date"].dt.strftime("%Y-%m-%d").tolist()
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle(
        f"Space Weather Go/No-Go Dashboard  |  {start_date} to {end_date}",
        fontsize=14,
        fontweight="bold",
    )

    axes[0].bar(dates, window["risk_score"], color=bar_colors, width=0.6)
    axes[0].axhline(
        20,
        color="#57606a",
        linestyle="--",
        linewidth=1,
        label="GO threshold (20)",
    )
    axes[0].axhline(
        60,
        color="#57606a",
        linestyle=":",
        linewidth=1,
        label="NO-GO threshold (60)",
    )
    axes[0].set_title("Risk Score per Day")
    axes[0].set_ylabel("Risk Score (0-100)")
    axes[0].set_ylim(0, 105)
    axes[0].legend(fontsize=8)
    axes[0].yaxis.grid(True, alpha=0.3)

    level_numbers = window["risk_level"].map(
        {"LOW": 1, "MODERATE": 2, "HIGH": 3, "EXTREME": 4}
    ).tolist()
    axes[1].bar(dates, level_numbers, color=bar_colors, width=0.6)
    axes[1].set_yticks([1, 2, 3, 4])
    axes[1].set_yticklabels(["GO", "CAUTION", "DELAY", "NO-GO"])
    axes[1].set_title("Daily Recommendation")
    axes[1].set_ylim(0, 5)
    axes[1].yaxis.grid(True, alpha=0.3)

    xclass = window["xclass_flare_count"]
    mclass = window["mclass_flare_count"]
    cclass = window["cclass_flare_count"]
    storms = window["storm_count"]
    axes[2].bar(dates, xclass, color="#dc2626", width=0.6, label="X-class")
    axes[2].bar(
        dates,
        mclass,
        bottom=xclass,
        color="#f97316",
        width=0.6,
        label="M-class",
    )
    axes[2].bar(
        dates,
        cclass,
        bottom=xclass + mclass,
        color="#eab308",
        width=0.6,
        label="C-class",
    )
    axes[2].bar(
        dates,
        storms,
        bottom=xclass + mclass + cclass,
        color="#7c5cd8",
        width=0.6,
        label="Storms",
    )
    axes[2].set_title("Solar Events in 48h Window")
    axes[2].set_xlabel("Date")
    axes[2].set_ylabel("Event Count")
    axes[2].legend(fontsize=8)
    axes[2].yaxis.grid(True, alpha=0.3)

    for axis in axes:
        axis.set_xticks(range(len(dates)))
        axis.set_xticklabels(dates, rotation=45, ha="right", fontsize=8)

    fig.tight_layout()
    if show:
        plt.show()
    return fig
