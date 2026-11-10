"""Flag unusual transactions with an Isolation Forest and a repeat-charge rule.

Run from the project root: python -m src.anomaly
"""

from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.predict import ROOT, extract_merchant

DATA_PATH = ROOT / "data" / "transactions.csv"
FLAGS_PATH = ROOT / "data" / "anomaly_flags.csv"
CONTAMINATION = 0.01
SEED = 42
REPEAT_WINDOW = timedelta(minutes=10)
NIGHT_HOURS = range(1, 6)
LARGE_RATIO = np.log(5)

# Amount relative to the merchant keeps large but normal payments, like rent, from being flagged.
# Day of week and merchant frequency were tried and lowered recall, so they are left out.
FOREST_FEATURES = ["amount_vs_merchant", "hour_share"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn transactions into the numeric features the detector uses."""
    timestamps = pd.to_datetime(df["timestamp"])
    merchants = df["description"].map(extract_merchant)
    amount = df["amount"].clip(lower=0)
    typical = amount.groupby(merchants).transform("median")
    hours = timestamps.dt.hour
    gaps = (
        df.assign(_ts=timestamps)
        .sort_values("_ts")
        .groupby(["description", "amount"])["_ts"]
        .diff()
    )
    return pd.DataFrame(
        {
            "amount_vs_merchant": np.log((amount + 1) / (typical + 1)),
            "hour_share": hours.map(hours.value_counts(normalize=True)),
            "hour": hours,
            "is_repeat": (gaps <= REPEAT_WINDOW).astype(int),
        },
        index=df.index,
    )


def _reason(row) -> str:
    if row["is_repeat"]:
        return "Possible duplicate charge"
    if row["hour"] in NIGHT_HOURS:
        return "Late-night charge"
    if row["amount_vs_merchant"] >= LARGE_RATIO:
        return "Unusually large for this merchant"
    return "Unusual pattern"


def flag_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with anomaly_score (higher is stranger), is_flagged and anomaly_reason."""
    features = build_features(df)
    forest_input = features[FOREST_FEATURES]
    forest = IsolationForest(n_estimators=300, contamination=CONTAMINATION, random_state=SEED).fit(forest_input)

    result = df.copy()
    result["anomaly_score"] = -forest.score_samples(forest_input)
    # An exact repeat looks ordinary to a statistical model, so a rule catches it.
    result["is_flagged"] = (forest.predict(forest_input) == -1) | (features["is_repeat"] == 1).to_numpy()
    result["anomaly_reason"] = ""
    flagged = result["is_flagged"]
    if flagged.any():
        result.loc[flagged, "anomaly_reason"] = features[flagged].apply(_reason, axis=1)
    return result


def evaluate(result: pd.DataFrame) -> dict:
    """Compare flags with the injected labels. Works only on generated data."""
    injected = result["is_anomaly"] == 1
    caught = result["is_flagged"] & injected
    by_type = caught[injected].groupby(result.loc[injected, "anomaly_type"]).mean().to_dict()
    return {
        "recall": float(caught.sum() / injected.sum()),
        "precision": float(caught.sum() / result["is_flagged"].sum()),
        "flagged": int(result["is_flagged"].sum()),
        "recall_by_type": {name: float(value) for name, value in by_type.items()},
    }


def main() -> None:
    if not DATA_PATH.exists():
        raise SystemExit(f"No data at {DATA_PATH}. Run: python -m src.generate_data")
    result = flag_anomalies(pd.read_csv(DATA_PATH))
    result.to_csv(FLAGS_PATH, index=False)

    metrics = evaluate(result)
    print(f"Flagged {metrics['flagged']} of {len(result)} transactions")
    print(f"Recall on injected anomalies: {metrics['recall']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    for name, recall in metrics["recall_by_type"].items():
        print(f"  {name:<14}{recall:.3f}")
    print(f"Saved flags to {FLAGS_PATH}")


if __name__ == "__main__":
    main()
