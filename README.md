# Retail Demand Forecasting & Inventory Intelligence

End-to-end Machine Learning project for **next-day retail demand forecasting**, covering the complete workflow from exploratory analysis and SQL to model deployment, CI/CD, monitoring, and data drift detection.

The system uses historical retail sales to predict demand for individual **store-item combinations**. Multiple approaches were evaluated using time-aware validation, with **XGBoost reducing MAE by 34.6% compared with a seasonal-naive baseline on the final untouched holdout set**.

---

## Key Results

| Model | MAE | RMSE | WAPE |
|---|---:|---:|---:|
| Seasonal Naive (lag 7) | 9.076 | 12.051 | 16.60% |
| XGBoost | **5.935** | **7.677** | **10.86%** |

**Final MAE improvement: 34.6%**

The final evaluation was performed on an untouched **Q4 2017 holdout set**.

---

## Project Overview

The objective is to simulate a production-oriented retail demand forecasting workflow.

The project includes:

- Exploratory Data Analysis
- Advanced SQL analysis with PostgreSQL
- Time-series feature engineering
- Leakage-safe lag and rolling features
- Seasonal forecasting baseline
- Random Forest modeling
- XGBoost modeling
- Time-based cross-validation
- Final holdout evaluation
- FastAPI inference service
- Automated testing with Pytest
- Docker containerization
- GitHub Actions CI pipeline
- GitHub Container Registry delivery
- Inference logging
- Basic data drift detection
- Interactive Streamlit dashboard

---

## System Architecture

```text
                         Retail Sales Data
                                |
                                v
                     PostgreSQL / SQL Analysis
                                |
                                v
                    Exploratory Data Analysis
                                |
                                v
                   Temporal Feature Engineering
                                |
              +-----------------+-----------------+
              |                                   |
              v                                   v
       Seasonal Baseline                    Machine Learning
          (lag 7)                        Random Forest / XGBoost
              |                                   |
              +-----------------+-----------------+
                                |
                                v
                    Time-Based Validation
                                |
                                v
                     Final Holdout Testing
                                |
                                v
                       Serialized Model
                                |
                                v
                            FastAPI
                                |
                                v
                             Docker
                                |
                                v
                       GitHub Actions CI
                         /           \
                        v             v
                     Pytest       Docker Build
                                      |
                                      v
                         GitHub Container Registry
                                      |
                                      v
                         Production Inference
                                      |
                         +------------+------------+
                         |                         |
                         v                         v
                  Prediction Logs          Drift Detection
```

---

## Dataset

The project uses the **Store Item Demand Forecasting Challenge** dataset.

The training dataset contains:

- **913,000 observations**
- **10 stores**
- **50 products**
- Daily observations
- Date range: **2013-01-01 to 2017-12-31**
- No missing values
- No duplicated observations

Main variables:

| Feature | Description |
|---|---|
| `date` | Observation date |
| `store` | Store identifier |
| `item` | Product identifier |
| `sales` | Daily units sold |

---

## Feature Engineering

Demand forecasting features are generated independently for each store-item combination.

### Calendar Features

- Year
- Month
- Day
- Day of week
- Week of year
- Quarter
- Weekend indicator

### Lag Features

```text
lag_1
lag_7
lag_14
lag_28
```

These capture recent and seasonal demand behavior.

### Rolling Statistics

```text
rolling_mean_7
rolling_mean_14
rolling_mean_28
rolling_std_7
rolling_std_28
```

All rolling features are calculated using:

```python
shift(1)
```

before the rolling operation.

This ensures that the current day's target is never included in the feature calculation and prevents **target leakage**.

---

## Validation Strategy

Random train/test splitting is inappropriate for forecasting because it allows future observations to influence training.

This project therefore uses chronological validation.

### Development Split

```text
Training
date < 2017-01-01

Validation
2017-01-01 <= date < 2017-10-01

Final Test
date >= 2017-10-01
```

The final Q4 2017 period remained untouched until model selection and development were completed.

---

## Temporal Cross-Validation

An expanding-window validation strategy was also implemented.

| Fold | Baseline MAE | XGBoost MAE | Improvement |
|---|---:|---:|---:|
| 2015 Q1 | 7.197 | 5.145 | 28.5% |
| 2016 Q1 | 7.494 | 5.285 | 29.5% |
| 2017 Q1 | 7.680 | 5.363 | 30.2% |

XGBoost consistently outperformed the seasonal baseline across all temporal folds.

This helps verify that the improvement is not dependent on a single validation period.

---

## Models

### Seasonal Naive Baseline

The baseline predicts demand using sales from the same store and product seven days earlier:

```text
prediction(t) = sales(t - 7)
```

This provides a meaningful benchmark for weekly retail seasonality.

### Random Forest

Random Forest was used as the first nonlinear ensemble model.

Validation performance:

```text
MAE   = 6.312
RMSE  = 8.216
WAPE  = 10.48%
```

### XGBoost

XGBoost produced the strongest validation performance.

Validation performance:

```text
MAE   = 6.156
RMSE  = 8.012
WAPE  = 10.22%
```

It was therefore selected as the final production model.

---

## Final Holdout Evaluation

After model selection, XGBoost was retrained using all available observations before October 2017.

The untouched Q4 2017 data was then used for the final evaluation.

```text
Seasonal Naive

MAE   = 9.076
RMSE  = 12.051
WAPE  = 16.60%

XGBoost

MAE   = 5.935
RMSE  = 7.677
WAPE  = 10.86%
```

### Final Result

**34.6% reduction in MAE compared with the seasonal baseline.**

---

## Forecasting Scope

The current model performs **next-day / one-step-ahead forecasting**.

Lag and rolling features use historical sales values that are assumed to be available at prediction time.

This is intentionally different from recursive multi-step forecasting.

A future extension could support multi-horizon forecasting using recursive predictions, direct horizon-specific models, or sequence models.

---

## PostgreSQL & SQL

The raw dataset was also loaded into PostgreSQL to reproduce a realistic analytical workflow.

SQL analysis includes:

- Sales aggregation by store and product
- Monthly and annual demand
- Common Table Expressions (CTEs)
- Window functions
- `LAG()`
- Month-over-month growth
- `RANK()`
- Top products by store

Example:

```sql
RANK() OVER (
    PARTITION BY store
    ORDER BY total_sales DESC
)
```

This part of the project demonstrates analytical SQL independently from the Python ML pipeline.

---

## REST API

The trained model is exposed through a **FastAPI** service.

### Health Check

```http
GET /health
```

Example response:

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### Prediction

```http
POST /predict
```

The endpoint validates incoming features using Pydantic and returns a non-negative demand prediction.

Example response:

```json
{
  "predicted_sales": 26.25
}
```

Interactive API documentation is automatically available through FastAPI Swagger UI.

---

## Automated Testing

The project includes automated tests for:

- API health
- Successful model inference
- Request validation
- WAPE calculation
- Lag feature leakage
- Rolling feature leakage

Run locally with:

```bash
pytest -v
```

Current test suite:

```text
7 passed
```

---

## Docker

The inference API is containerized with Docker.

Build locally:

```bash
docker build -t retail-demand-api .
```

Run:

```bash
docker run -p 8000:8000 retail-demand-api
```

The API is then available at:

```text
http://localhost:8000
```

---

## CI/CD

GitHub Actions automatically executes the pipeline when code is pushed to the main branch.

```text
Git Push
   |
   v
Automated Tests
   |
   v
Docker Build
   |
   v
Publish Docker Image
   |
   v
GitHub Container Registry
```

The pipeline prevents container publication when the automated test stage fails.

The resulting container image can be pulled from GitHub Container Registry:

```bash
docker pull ghcr.io/0311869uaslp-a11y/retail-demand-forecasting:latest
```

This provides a reproducible and version-controlled delivery workflow for the ML inference service.

---

## Model Monitoring

Production inference requests are logged using JSON Lines.

Each record includes:

```text
timestamp
model_version
input features
prediction
```

Example:

```json
{
  "timestamp": "2026-09-29T00:18:03+00:00",
  "model_version": "1.0.0",
  "features": {
    "store": 1,
    "item": 1,
    "lag_1": 22.0,
    "lag_7": 24.0
  },
  "prediction": 26.25
}
```

This creates an inference history that can later be combined with observed sales to monitor model performance.

---

## Data Drift Detection

A lightweight drift-monitoring component compares selected production features with the historical training reference distribution.

Currently monitored features include:

```text
lag_1
lag_7
rolling_mean_7
rolling_mean_28
rolling_std_28
```

The current implementation calculates normalized mean shift:

```text
| production_mean - reference_mean |
------------------------------------
         reference_std
```

A threshold is used to flag substantial distribution changes.

This is intentionally a lightweight monitoring implementation. A production extension could incorporate PSI, Kolmogorov-Smirnov tests, Evidently, alerting, and automated retraining policies.

---

## Interactive Dashboard

A Streamlit dashboard provides a business-oriented view of model performance.

It includes:

- Final holdout KPIs
- MAE
- RMSE
- WAPE
- Improvement versus seasonal baseline
- Actual vs. predicted demand
- Store selection
- Product selection
- Store-level model performance

Run locally with:

```bash
streamlit run dashboard/app.py
```

---

## Project Structure

```text
retail-demand-forecasting/
|
|-- dashboard/
|   `-- app.py
|
|-- data/
|   |-- raw/
|   `-- processed/
|
|-- models/
|   `-- xgboost_demand_forecaster.joblib
|
|-- notebooks/
|   |-- 01_exploratory_data_analysis.ipynb
|   `-- 02_feature_engineering.ipynb
|
|-- sql/
|   `-- 01_exploratory_analysis.sql
|
|-- src/
|   |-- api.py
|   |-- drift.py
|   |-- features.py
|   |-- metrics.py
|   |-- monitoring.py
|   `-- train.py
|
|-- tests/
|   |-- test_api.py
|   `-- test_features.py
|
|-- .github/
|   `-- workflows/
|       `-- ci.yml
|
|-- Dockerfile
|-- pytest.ini
|-- requirements.txt
|-- requirements-dev.txt
|-- requirements-test.txt
`-- README.md
```

---

## Tech Stack

**Machine Learning**

`Python` · `pandas` · `NumPy` · `scikit-learn` · `XGBoost`

**Data & Analytics**

`PostgreSQL` · `SQL` · `Time Series` · `Feature Engineering`

**Backend & Deployment**

`FastAPI` · `Pydantic` · `Docker`

**MLOps**

`Pytest` · `GitHub Actions` · `GitHub Container Registry` · `Model Monitoring` · `Data Drift Detection`

**Visualization**

`Matplotlib` · `Streamlit`

---

## Limitations

The dataset does not contain several variables that would normally influence retail demand, including:

- Product prices
- Promotions
- Holidays
- Inventory availability
- Marketing campaigns
- Competitor information

Therefore, the project focuses primarily on extracting predictive information from historical demand patterns.

The current API also expects engineered lag and rolling features as input. In a production architecture, these features would typically be generated by an upstream feature pipeline or retrieved from a feature store.

---

## Future Improvements

Potential extensions include:

- Automated model retraining
- Prediction performance monitoring after actual demand becomes available
- PSI / statistical drift tests
- MLflow model registry
- Cloud deployment
- Multi-step demand forecasting
- Promotion and price features
- Inventory optimization and reorder-point recommendations
- Deep-learning forecasting experiments

---

## Author

**José Luis Romero Vázquez**

Electronic Engineer with an international double Master's background in Electronic Engineering, Computer Networks and Telecommunications, with experience in Machine Learning, IoT, data analysis, research, and software development.

Mexico · France · Lithuania