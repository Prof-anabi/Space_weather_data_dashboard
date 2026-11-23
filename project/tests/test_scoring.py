import pandas as pd

from space_weather.scoring import calculate_risk_score, categorize_risk


def test_risk_score_is_capped_at_100():
    row = pd.Series(
        {
            "xclass_flare_count": 3,
            "mclass_flare_count": 3,
            "max_kp_index": 9,
            "event_trend": 3,
        }
    )
    assert calculate_risk_score(row) == 100.0


def test_risk_categories():
    assert categorize_risk(20) == "LOW"
    assert categorize_risk(40) == "MODERATE"
    assert categorize_risk(60) == "HIGH"
    assert categorize_risk(61) == "EXTREME"
