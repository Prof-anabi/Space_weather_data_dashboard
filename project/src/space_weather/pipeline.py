from pathlib import Path

from .artifacts import save_artifacts
from .data_loader import load_csv
from .exploration import plot_exploration, print_exploration_summary
from .features import build_risk_features
from .model import FEATURE_COLUMNS, train_model
from .preprocessing import clean_data
from .scoring import add_risk_columns
from .visualization import (
    plot_dashboard,
    print_window_summary,
    summarize_window,
)


def run_pipeline(
    project_directory: str | Path,
    start_date: str = "2025-05-01",
    end_date: str = "2025-07-09",
    show_dashboard: bool = True,
) -> dict:
    project_dir = Path(project_directory).resolve()
    csv_path = project_dir / "data" / "space_weather_unified.csv"

    raw = load_csv(csv_path)
    cleaned = clean_data(raw)
    print_exploration_summary(cleaned)
    plot_exploration(cleaned, show=False)

    features = build_risk_features(cleaned)
    scored = add_risk_columns(features)

    training = train_model(scored)
    print(f"\nTest accuracy: {training.accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(training.report)
    print("\nFeature Importances:")
    for name, importance in training.feature_importances.items():
        print(f"  {name:<25} {importance:.4f}")

    model_path, data_path = save_artifacts(
        training.model,
        scored,
        FEATURE_COLUMNS,
        project_dir / "models",
    )
    print(f"\nModel saved to: {model_path}")
    print(f"Data saved to: {data_path}")

    window, summary = summarize_window(scored, start_date, end_date)
    print_window_summary(summary, start_date, end_date)
    plot_dashboard(window, start_date, end_date, show=show_dashboard)

    return {
        "cleaned": cleaned,
        "risk_features": scored,
        "training": training,
        "model_path": model_path,
        "data_path": data_path,
        "window": window,
        "summary": summary,
    }
