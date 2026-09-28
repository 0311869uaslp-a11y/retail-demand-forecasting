from fastapi.testclient import TestClient

from src.api import app


client = TestClient(app)


def test_health_endpoint():

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    

def test_prediction_endpoint():

    payload = {
        "store": 1,
        "item": 1,
        "year": 2017,
        "month": 10,
        "day": 1,
        "day_of_week": 6,
        "week_of_year": 39,
        "quarter": 4,
        "is_weekend": 1,
        "lag_1": 22,
        "lag_7": 24,
        "lag_14": 21,
        "lag_28": 23,
        "rolling_mean_7": 23.4,
        "rolling_mean_14": 22.8,
        "rolling_mean_28": 23.1,
        "rolling_std_7": 4.2,
        "rolling_std_28": 5.1,
    }

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 200

    data = response.json()

    assert "predicted_sales" in data
    assert isinstance(
        data["predicted_sales"],
        (int, float)
    )

    assert data["predicted_sales"] >= 0
    
def test_invalid_month():

    payload = {
        "store": 1,
        "item": 1,
        "year": 2017,

        # Invalid
        "month": 25,

        "day": 1,
        "day_of_week": 6,
        "week_of_year": 39,
        "quarter": 4,
        "is_weekend": 1,
        "lag_1": 22,
        "lag_7": 24,
        "lag_14": 21,
        "lag_28": 23,
        "rolling_mean_7": 23.4,
        "rolling_mean_14": 22.8,
        "rolling_mean_28": 23.1,
        "rolling_std_7": 4.2,
        "rolling_std_28": 5.1,
    }

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 422
    
import numpy as np

from src.metrics import wape


def test_wape_perfect_prediction():

    y_true = np.array([
        10,
        20,
        30
    ])

    y_pred = np.array([
        10,
        20,
        30
    ])

    result = wape(
        y_true,
        y_pred
    )

    assert result == 0

def test_wape_known_value():

    y_true = np.array([
        100,
        100
    ])

    y_pred = np.array([
        90,
        110
    ])

    result = wape(
        y_true,
        y_pred
    )

    assert result == 10


