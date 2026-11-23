from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


FEATURE_COLUMNS = [
    "xclass_flare_count",
    "mclass_flare_count",
    "cclass_flare_count",
    "max_kp_index",
    "avg_kp_index",
    "storm_count",
    "event_trend",
]


@dataclass
class TrainingResult:
    model: RandomForestClassifier
    accuracy: float
    report: str
    feature_importances: pd.Series


def train_model(
    risk_features: pd.DataFrame,
    cutoff: str = "2025-01-01",
) -> TrainingResult:
    cutoff_timestamp = pd.Timestamp(cutoff)
    train_mask = risk_features["date"] < cutoff_timestamp
    test_mask = risk_features["date"] >= cutoff_timestamp

    x_train = risk_features.loc[train_mask, FEATURE_COLUMNS]
    x_test = risk_features.loc[test_mask, FEATURE_COLUMNS]
    y_train = risk_features.loc[train_mask, "risk_level"]
    y_test = risk_features.loc[test_mask, "risk_level"]

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    importances = pd.Series(
        model.feature_importances_,
        index=FEATURE_COLUMNS,
    ).sort_values(ascending=False)

    return TrainingResult(
        model=model,
        accuracy=accuracy_score(y_test, predictions),
        report=classification_report(y_test, predictions, zero_division=0),
        feature_importances=importances,
    )
