import matplotlib.pyplot as plt
import pandas as pd


def print_exploration_summary(data: pd.DataFrame) -> None:
    event_counts = data["event_type"].value_counts()
    event_summary = pd.DataFrame(
        {
            "Count": event_counts,
            "Percentage": (event_counts / len(data) * 100).round(2),
        }
    )
    print("=== 1. Event Type Distribution ===")
    print(event_summary.to_string())

    print("\n=== 2. Temporal Summary ===")
    print("Events per Year:")
    print(data["year"].value_counts().sort_index().to_string())
    print("\nEvents per Month:")
    print(data["month"].value_counts().sort_index().to_string())
    print("\nEvents per Hour:")
    print(data["hour"].value_counts().sort_index().to_string())


def plot_exploration(data: pd.DataFrame, show: bool = True) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Exploratory Data Analysis", fontsize=15, fontweight="bold")

    event_counts = data["event_type"].value_counts().sort_values(ascending=True)
    bars = axes[0, 0].barh(event_counts.index, event_counts.values, color="#7c5cd8")
    for bar, value in zip(bars, event_counts.values):
        axes[0, 0].text(
            bar.get_width() + event_counts.values.max() * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{value:,}",
            va="center",
            fontsize=9,
        )
    axes[0, 0].set_title("Event Type Distribution")
    axes[0, 0].set_xlabel("Count")

    monthly = data["month"].value_counts().sort_index()
    axes[0, 1].bar(monthly.index, monthly.values, color="#3b82d4")
    axes[0, 1].set_xticks(range(1, 13))
    axes[0, 1].yaxis.grid(True, alpha=0.3)
    axes[0, 1].set_title("Events per Month")
    axes[0, 1].set_xlabel("Month")
    axes[0, 1].set_ylabel("Count")

    flare_data = data[
        (data["event_type"] == "Solar Flare")
        & (data["flare_class"] != "N/A")
    ]
    flare_counts = flare_data["flare_class"].value_counts().sort_index()
    colors = {
        "X": "#dc2626",
        "M": "#f97316",
        "C": "#eab308",
        "B": "#22c55e",
    }
    bars = axes[1, 0].bar(
        flare_counts.index,
        flare_counts.values,
        color=[colors.get(value, "#94a3b8") for value in flare_counts.index],
    )
    for bar, value in zip(bars, flare_counts.values):
        axes[1, 0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + flare_counts.values.max() * 0.01,
            f"{value:,}",
            ha="center",
            fontsize=9,
        )
    axes[1, 0].set_title("Solar Flare Class Breakdown")
    axes[1, 0].set_xlabel("Flare Class")
    axes[1, 0].set_ylabel("Count")

    yearly = data["year"].value_counts().sort_index()
    bars = axes[1, 1].bar(
        yearly.index.astype(str),
        yearly.values,
        color="#10b981",
    )
    for bar, value in zip(bars, yearly.values):
        axes[1, 1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + yearly.values.max() * 0.01,
            f"{value:,}",
            ha="center",
            fontsize=9,
        )
    axes[1, 1].set_title("Events per Year")
    axes[1, 1].set_xlabel("Year")
    axes[1, 1].set_ylabel("Count")

    fig.tight_layout()
    if show:
        plt.show()
    return fig
