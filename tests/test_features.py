import pandas as pd

from src.features import create_features


def test_lag_features_do_not_use_current_target():

    df = pd.DataFrame({
        "date": pd.date_range(
            start="2026-01-01",
            periods=35,
            freq="D"
        ),
        "store": [1] * 35,
        "item": [1] * 35,
        "sales": list(range(1, 36))
    })

    result = create_features(df)

    row = result.iloc[28]

    assert row["sales"] == 29
    assert row["lag_1"] == 28
    assert row["lag_7"] == 22
    assert row["lag_14"] == 15
    assert row["lag_28"] == 1


def test_rolling_mean_excludes_current_day():

    df = pd.DataFrame({
        "date": pd.date_range(
            start="2026-01-01",
            periods=10,
            freq="D"
        ),
        "store": [1] * 10,
        "item": [1] * 10,
        "sales": [
            10,
            20,
            30,
            40,
            50,
            60,
            70,
            1000,
            1000,
            1000
        ]
    })

    result = create_features(df)

    row = result.iloc[7]

    expected_mean = (
        10 + 20 + 30 + 40 + 50 + 60 + 70
    ) / 7

    assert row["rolling_mean_7"] == expected_mean