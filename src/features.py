import pandas as pd


FEATURES = [
    "store",
    "item",
    "year",
    "month",
    "day",
    "day_of_week",
    "week_of_year",
    "quarter",
    "is_weekend",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_28",
    "rolling_std_7",
    "rolling_std_28",
]


TARGET = "sales"


def create_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    # Ensure chronological order
    df = df.sort_values(
        ["store", "item", "date"]
    ).reset_index(drop=True)

    # Calendar features
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek

    df["week_of_year"] = (
        df["date"]
        .dt
        .isocalendar()
        .week
        .astype(int)
    )

    df["quarter"] = df["date"].dt.quarter

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    # Sales history by store/item
    group = df.groupby(
        ["store", "item"]
    )["sales"]

    # Lag features
    df["lag_1"] = group.shift(1)
    df["lag_7"] = group.shift(7)
    df["lag_14"] = group.shift(14)
    df["lag_28"] = group.shift(28)

    # Rolling features
    df["rolling_mean_7"] = group.transform(
        lambda x: x.shift(1).rolling(7).mean()
    )

    df["rolling_mean_14"] = group.transform(
        lambda x: x.shift(1).rolling(14).mean()
    )

    df["rolling_mean_28"] = group.transform(
        lambda x: x.shift(1).rolling(28).mean()
    )

    df["rolling_std_7"] = group.transform(
        lambda x: x.shift(1).rolling(7).std()
    )

    df["rolling_std_28"] = group.transform(
        lambda x: x.shift(1).rolling(28).std()
    )

    return df