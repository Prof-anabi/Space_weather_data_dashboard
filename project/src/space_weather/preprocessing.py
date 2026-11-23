import numpy as np
import pandas as pd


def extract_flare_info(row: pd.Series) -> pd.Series:
    if (
        row["event_type"] == "Solar Flare"
        and pd.notna(row["class_type"])
        and row["class_type"] != "Unknown"
    ):
        class_str = str(row["class_type"])
        flare_class = class_str[0] if class_str else "N/A"
        magnitude_text = "".join(
            character
            for character in class_str
            if character.isdigit() or character == "."
        )
        try:
            magnitude = float(magnitude_text) if magnitude_text else 0.0
        except ValueError:
            magnitude = 0.0
        return pd.Series(
            {"flare_magnitude": magnitude, "flare_class": flare_class}
        )
    return pd.Series({"flare_magnitude": 0.0, "flare_class": "N/A"})


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    cleaned = data.copy()

    for column, value in {
        "kp_index": 0.0,
        "class_type": "Unknown",
        "source_location": "Unknown",
        "active_region": "Unknown",
        "note": "",
    }.items():
        cleaned[column] = cleaned[column].fillna(value)

    cleaned = cleaned.drop_duplicates(subset=["event_id"], keep="first")

    for column in ["begin_time", "peak_time", "end_time", "observed_time"]:
        cleaned[column] = (
            pd.to_datetime(cleaned[column], errors="coerce", utc=True)
            .dt.tz_convert(None)
        )

    cleaned["year"] = (
        cleaned["begin_time"].dt.year.fillna(0).astype(int)
    )
    cleaned["month"] = (
        cleaned["begin_time"].dt.month.fillna(0).astype(int)
    )
    cleaned["hour"] = (
        cleaned["begin_time"].dt.hour.fillna(0).astype(int)
    )
    cleaned["duration_minutes"] = (
        (cleaned["end_time"] - cleaned["begin_time"])
        .dt.total_seconds()
        .div(60)
        .fillna(0.0)
    )
    cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce")

    cleaned[["flare_magnitude", "flare_class"]] = cleaned.apply(
        extract_flare_info,
        axis=1,
    )
    return cleaned.replace({np.nan: None})
