from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)

from xgboost import XGBRegressor

from src.features import (
    create_features,
    FEATURES,
    TARGET,
)

from src.metrics import wape

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "train.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "xgboost_demand_forecaster.joblib"
)

def load_data():

    print("Loading data...")

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["date"]
    )

    print(
        f"Loaded {len(df):,} observations."
    )

    return df

def train_model(X, y):

    print("Training XGBoost model...")

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        n_jobs=-1,
        random_state=42,
    )

    model.fit(X, y)

    return model

def evaluate_model(
    model,
    X_test,
    y_test
):

    predictions = model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    model_wape = wape(
        y_test,
        predictions
    )

    print("\nFinal Test Performance")
    print("----------------------")
    print(f"MAE:  {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"WAPE: {model_wape:.2f}%")

    return predictions

def main():

    # Load
    df = load_data()

    # Feature engineering
    print("Creating features...")

    df = create_features(df)

    df = df.dropna().copy()

    # Time-based split
    train = df[
        df["date"] < "2017-10-01"
    ].copy()

    test = df[
        df["date"] >= "2017-10-01"
    ].copy()

    print(
        f"Training observations: {len(train):,}"
    )

    print(
        f"Test observations: {len(test):,}"
    )

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]

    # Train
    model = train_model(
        X_train,
        y_train
    )

    # Evaluate
    evaluate_model(
        model,
        X_test,
        y_test
    )

    # Save
    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_PATH
    )

    print(
        f"\nModel saved to:\n{MODEL_PATH}"
    )


if __name__ == "__main__":
    main()