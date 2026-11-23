from datetime import timedelta

import pandas as pd


def build_risk_features(
    data: pd.DataFrame,
    start_date: str = "2023-01-01",
) -> pd.DataFrame:
    filtered = data.loc[data["date"] >= pd.Timestamp(start_date)].copy()
    filtered = filtered.sort_values("date").reset_index(drop=True)

    rows = []
    for current_date in filtered["date"].drop_duplicates():
        window_start = current_date - timedelta(hours=48)
        window_data = filtered[
            (filtered["date"] >= window_start)
            & (filtered["date"] < current_date)
        ]

        kp_values = window_data["kp_index"].dropna()
        last_24h = filtered[
            (filtered["date"] >= current_date - timedelta(hours=24))
            & (filtered["date"] < current_date)
        ]
        previous_24h = filtered[
            (filtered["date"] >= current_date - timedelta(hours=48))
            & (
                filtered["date"]
                < current_date - timedelta(hours=24)
            )
        ]

        trend = (
            len(last_24h) / len(previous_24h)
            if len(previous_24h) > 0
            else 1.0
        )
        rows.append(
            {
                "date": current_date,
                "xclass_flare_count": int(
                    (
                        (window_data["event_type"] == "Solar Flare")
                        & (window_data["flare_class"] == "X")
                    ).sum()
                ),
                "mclass_flare_count": int(
                    (
                        (window_data["event_type"] == "Solar Flare")
                        & (window_data["flare_class"] == "M")
                    ).sum()
                ),
                "cclass_flare_count": int(
                    (
                        (window_data["event_type"] == "Solar Flare")
                        & (window_data["flare_class"] == "C")
                    ).sum()
                ),
                "max_kp_index": float(kp_values.max()) if len(kp_values) else 0.0,
                "avg_kp_index": float(kp_values.mean()) if len(kp_values) else 0.0,
                "storm_count": int((window_data["kp_index"] >= 5).sum()),
                "event_trend": float(trend),
            }
        )
    return pd.DataFrame(rows)
