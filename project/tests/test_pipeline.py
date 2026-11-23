from pathlib import Path

import pandas as pd

from space_weather.features import build_risk_features
from space_weather.preprocessing import clean_data


def test_feature_builder_excludes_current_date():
    data = pd.DataFrame(
        [
            {
                "event_id": "1",
                "event_type": "Solar Flare",
                "begin_time": "2025-01-01T00:00:00Z",
                "peak_time": None,
                "end_time": None,
                "observed_time": None,
                "date": "2025-01-01",
                "year": 2025,
                "month": 1,
                "hour": 0,
                "kp_index": 6,
                "class_type": "M5.0",
                "source_location": "Unknown",
                "active_region": "1",
                "note": "",
            },
            {
                "event_id": "2",
                "event_type": "Solar Flare",
                "begin_time": "2025-01-02T00:00:00Z",
                "peak_time": None,
                "end_time": None,
                "observed_time": None,
                "date": "2025-01-02",
                "year": 2025,
                "month": 1,
                "hour": 0,
                "kp_index": 2,
                "class_type": "C1.0",
                "source_location": "Unknown",
                "active_region": "1",
                "note": "",
            },
        ]
    )
    cleaned = clean_data(data)
    features = build_risk_features(cleaned)
    assert len(features) == 2
    assert features.iloc[1]["mclass_flare_count"] == 1
