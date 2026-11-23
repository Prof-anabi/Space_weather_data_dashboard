import pandas as pd


def calculate_risk_score(row: pd.Series) -> float:
    x_score = min(row["xclass_flare_count"] * 40, 40)
    m_score = min(row["mclass_flare_count"] * 25, 25)
    kp_score = (row["max_kp_index"] / 9) * 20
    trend_score = min(max((row["event_trend"] - 1) * 15, 0), 15)
    return float(min(x_score + m_score + kp_score + trend_score, 100))


def categorize_risk(score: float) -> str:
    if score <= 20:
        return "LOW"
    if score <= 40:
        return "MODERATE"
    if score <= 60:
        return "HIGH"
    return "EXTREME"


def add_risk_columns(features: pd.DataFrame) -> pd.DataFrame:
    scored = features.copy()
    scored["risk_score"] = scored.apply(calculate_risk_score, axis=1)
    scored["risk_level"] = scored["risk_score"].apply(categorize_risk)
    return scored
