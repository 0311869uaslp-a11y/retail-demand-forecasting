from pathlib import Path
import sys

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.features import create_features, FEATURES
from src.metrics import wape


DATA_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_demand_forecaster.joblib"


st.set_page_config(
    page_title="Retail Demand Forecasting",
    page_icon="📈",
    layout="wide",
)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = create_features(df).dropna()

    return df[df["date"] >= "2017-10-01"].copy()


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


df = load_data()
model = load_model()

df["prediction"] = model.predict(df[FEATURES])
df["prediction"] = df["prediction"].clip(lower=0)

df["baseline_prediction"] = df["lag_7"]

mae = mean_absolute_error(df["sales"], df["prediction"])
rmse = mean_squared_error(df["sales"], df["prediction"]) ** 0.5
model_wape = wape(df["sales"], df["prediction"])

baseline_mae = mean_absolute_error(
    df["sales"],
    df["baseline_prediction"],
)

improvement = (baseline_mae - mae) / baseline_mae * 100


st.title("Retail Demand Forecasting & Inventory Intelligence")

st.caption(
    "End-to-end machine learning system for next-day retail demand forecasting "
    "using XGBoost, temporal feature engineering and production-oriented MLOps."
)

st.subheader("Final Holdout Performance")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "MAE",
    f"{mae:.2f}",
)

col2.metric(
    "RMSE",
    f"{rmse:.2f}",
)

col3.metric(
    "WAPE",
    f"{model_wape:.2f}%",
)

col4.metric(
    "MAE Improvement",
    f"{improvement:.1f}%",
    delta="vs seasonal baseline",
)


st.divider()

st.subheader("Demand Forecast")

store = st.selectbox(
    "Store",
    sorted(df["store"].unique()),
)

item = st.selectbox(
    "Item",
    sorted(df["item"].unique()),
)

filtered = df[
    (df["store"] == store)
    & (df["item"] == item)
].copy()

chart_data = (
    filtered[
        ["date", "sales", "prediction"]
    ]
    .set_index("date")
    .rename(
        columns={
            "sales": "Actual Sales",
            "prediction": "XGBoost Forecast",
        }
    )
)

st.line_chart(chart_data)


st.divider()

st.subheader("Performance by Store")

store_performance = (
    df.groupby("store")
    .apply(
        lambda x: pd.Series(
            {
                "MAE": mean_absolute_error(
                    x["sales"],
                    x["prediction"],
                ),
                "WAPE": wape(
                    x["sales"],
                    x["prediction"],
                ),
            }
        ),
        include_groups=False,
    )
    .reset_index()
)

st.dataframe(
    store_performance.style.format(
        {
            "MAE": "{:.2f}",
            "WAPE": "{:.2f}%",
        }
    ),
    use_container_width=True,
)


st.divider()

st.subheader("Project Architecture")

st.code(
    """
Raw Retail Data
       |
       v
PostgreSQL / SQL Analysis
       |
       v
Temporal Feature Engineering
       |
       v
Time-Based Validation
       |
       v
Seasonal Baseline ---> XGBoost
                         |
                         v
                     FastAPI
                         |
                         v
                      Docker
                         |
                         v
                   GitHub Actions
                         |
                         v
                       GHCR
                         |
                         v
              Monitoring & Drift Detection
"""
)