from pathlib import Path
import json

import numpy as np
import pandas as pd

from src.features import create_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"
LOG_PATH = PROJECT_ROOT / "logs" / "predictions.jsonl"

MONITORED_FEATURES = [
    "lag_1",
    "lag_7",
    "rolling_mean_7",
    "rolling_mean_28",
    "rolling_std_28",
]


def load_reference_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = create_features(df).dropna()

    # Same historical period used before the final holdout
    reference = df[df["date"] < "2017-10-01"]

    return reference


def load_production_data():
    records = []

    if not LOG_PATH.exists():
        return pd.DataFrame()

    with LOG_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            records.append(record["features"])

    return pd.DataFrame(records)


def calculate_drift(reference, production):
    results = []

    for feature in MONITORED_FEATURES:
        ref_mean = reference[feature].mean()
        ref_std = reference[feature].std()

        prod_mean = production[feature].mean()

        normalized_shift = abs(prod_mean - ref_mean) / max(ref_std, 1e-8)

        drift_detected = normalized_shift > 1.0

        results.append({
            "feature": feature,
            "reference_mean": round(ref_mean, 2),
            "production_mean": round(prod_mean, 2),
            "normalized_shift": round(normalized_shift, 3),
            "drift_detected": drift_detected,
        })

    return pd.DataFrame(results)


def main():
    reference = load_reference_data()
    production = load_production_data()

    if production.empty:
        print("No production predictions available.")
        return

    results = calculate_drift(reference, production)

    print("\nMODEL DATA DRIFT REPORT")
    print("=" * 70)
    print(results.to_string(index=False))

    drift_count = results["drift_detected"].sum()

    print("\nSummary:")
    print(f"Features monitored: {len(results)}")
    print(f"Features with detected drift: {drift_count}")

    if drift_count > 0:
        print("Status: DRIFT DETECTED")
    else:
        print("Status: NO SIGNIFICANT DRIFT")


if __name__ == "__main__":
    main()