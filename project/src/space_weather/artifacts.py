from pathlib import Path

import joblib
import pandas as pd


def recommendation_for(level: str) -> str:
    return {
        "LOW": "GO",
        "MODERATE": "CAUTION",
        "HIGH": "DELAY",
        "EXTREME": "NO-GO",
    }.get(level, "NO-GO")


def save_artifacts(
    model,
    risk_features: pd.DataFrame,
    feature_columns: list[str],
    output_directory: str | Path,
) -> tuple[Path, Path]:
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    latest = risk_features.iloc[-1]

    current_stats = {
        "latest_date": latest["date"],
        "latest_risk_score": latest["risk_score"],
        "latest_risk_level": latest["risk_level"],
        "xclass_48h": int(latest["xclass_flare_count"]),
        "mclass_48h": int(latest["mclass_flare_count"]),
        "max_kp_48h": float(latest["max_kp_index"]),
        "recommendation": recommendation_for(latest["risk_level"]),
    }
    data_package = {
        "current_stats": current_stats,
        "feature_cols": feature_columns,
        "risk_features": risk_features.to_dict("records"),
    }

    model_path = directory / "launch_decision_model.pkl"
    data_path = directory / "space_weather_data.pkl"
    joblib.dump(model, model_path)
    joblib.dump(data_package, data_path)
    return model_path, data_path
