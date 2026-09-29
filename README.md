# Retail Demand Forecasting & Inventory Intelligence

![Python](https://img.shields.io/badge/Python-3.13-blue)
![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange)
![FastAPI](https://img.shields.io/badge/API-FastAPI-green)
![Docker](https://img.shields.io/badge/Container-Docker-blue)
![Azure](https://img.shields.io/badge/Cloud-Azure-blue)
![CI](https://img.shields.io/badge/CI-GitHub%20Actions-black)
![Tests](https://img.shields.io/badge/tests-7%20passed-brightgreen)

End-to-end Machine Learning project for **next-day retail demand forecasting**, covering the complete workflow from exploratory data analysis and advanced SQL to model training, temporal validation, REST API development, automated testing, containerization, CI/CD, monitoring, drift detection, and cloud deployment on Microsoft Azure.

The system uses historical retail sales to predict demand for individual **store-item combinations**. Multiple approaches were evaluated using leakage-safe, time-aware validation, with **XGBoost reducing MAE by 34.6% compared with a seasonal-naive baseline on the final untouched Q4 2017 holdout set**.

The trained model is productionized through **FastAPI and Docker**, automatically tested and published through **GitHub Actions and GitHub Container Registry (GHCR)**, and deployed as a publicly accessible HTTPS inference service using **Microsoft Azure Container Apps**.

---

## Key Results

| Model | MAE | RMSE | WAPE |
|---|---:|---:|---:|
| Seasonal Naive (lag 7) | 9.076 | 12.051 | 16.60% |
| XGBoost | **5.935** | **7.677** | **10.86%** |

**Final MAE improvement: 34.6%**

The final evaluation was performed on an untouched **Q4 2017 temporal holdout set**.

---

## Project Overview

The objective is to simulate a production-oriented retail demand forecasting workflow that goes beyond model experimentation and covers the main stages required to transform a predictive model into a deployable ML service.

The project includes:

- Exploratory Data Analysis
- Advanced SQL analysis with PostgreSQL
- Time-series feature engineering
- Leakage-safe lag and rolling features
- Seasonal forecasting baseline
- Random Forest modeling
- XGBoost modeling
- Time-based cross-validation
- Final temporal holdout evaluation
- FastAPI inference service
- Automated testing with Pytest
- Docker containerization
- GitHub Actions CI pipeline
- GitHub Container Registry delivery
- Microsoft Azure Container Apps deployment
- Public HTTPS inference endpoint
- Scale-to-zero cloud configuration
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
                         Microsoft Azure
                         Container Apps
                                      |
                                      v
                          Public HTTPS API
                           /          \
                          v            v
                    GET /health    POST /predict
                                      |
                                      v
                              XGBoost Inference
                                      |
                         +------------+------------+
                         |                         |
                         v                         v
                  Prediction Logs           Drift Detection
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

These variables capture recent and seasonal demand behavior.

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

Random train/test splitting is inappropriate for forecasting because it can allow future observations to influence model development.

This project therefore uses **chronological validation**.

### Development Split

```text
Training
date < 2017-01-01

Validation
2017-01-01 <= date < 2017-10-01

Final Test
date >= 2017-10-01
```

The final **Q4 2017** period remained untouched until model selection and development were completed.

This provides a more realistic estimate of model performance on future observations.

---

## Temporal Cross-Validation

An **expanding-window validation strategy** was also implemented to evaluate model stability across different temporal periods.

| Fold | Baseline MAE | XGBoost MAE | Improvement |
|---|---:|---:|---:|
| 2015 Q1 | 7.197 | 5.145 | 28.5% |
| 2016 Q1 | 7.494 | 5.285 | 29.5% |
| 2017 Q1 | 7.680 | 5.363 | 30.2% |

XGBoost consistently outperformed the seasonal baseline across all evaluated temporal folds.

This helps verify that the observed improvement is not dependent on a single validation period.

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

**34.6% reduction in MAE compared with the seasonal-naive baseline.**

The final model also reduced RMSE from **12.051 to 7.677**, demonstrating a substantial improvement over the weekly seasonal benchmark on previously unseen future data.

---

## Forecasting Scope

The current model performs **next-day / one-step-ahead forecasting**.

Lag and rolling features use historical sales values that are assumed to be available at prediction time.

This is intentionally different from recursive multi-step forecasting.

The current architecture is therefore designed to answer a question such as:

> Given the latest historical demand information for a store-item combination, what is the expected demand for the next day?

Future versions could support multi-horizon forecasting using recursive predictions, direct horizon-specific models, or sequence-based forecasting approaches.

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

The trained XGBoost model is exposed through a **FastAPI REST service**.

The serialized model is loaded when the API starts and incoming prediction requests are validated using **Pydantic**.

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

The endpoint validates incoming features and returns a non-negative demand prediction.

Example response:

```json
{
  "predicted_sales": 26.25
}
```

Interactive API documentation is automatically available through **FastAPI Swagger UI**.

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

The tests are also executed automatically through GitHub Actions before successful container builds are published.

---

## Docker

The inference API is containerized with **Docker**, providing a reproducible environment for local and cloud execution.

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

Health check:

```text
http://localhost:8000/health
```

Interactive documentation:

```text
http://localhost:8000/docs
```

---

## CI/CD

GitHub Actions automatically executes the CI pipeline when code is pushed to the main branch.

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

The CI pipeline automatically validates the application, builds the Docker image, and publishes successful builds to GHCR. The published container image is then used as the deployment artifact for the **Azure Container Apps** inference service.

> **Note:** The current CI pipeline automatically tests, builds, and publishes the container image. Deployment from GHCR to Azure Container Apps is currently performed separately rather than as an automated continuous-deployment stage.

---

## Cloud Deployment — Microsoft Azure

The containerized inference service is deployed to **Microsoft Azure Container Apps**.

The production container image is stored in **GitHub Container Registry (GHCR)** and deployed to Azure as a containerized FastAPI inference service running the trained XGBoost forecasting model.

### Deployment Architecture

```text
GitHub Repository
       |
       v
GitHub Actions
       |
       +------> Automated Tests (Pytest)
       |
       v
Docker Build
       |
       v
GitHub Container Registry
       |
       v
Azure Container Apps
       |
       v
FastAPI + XGBoost
       |
       v
Public HTTPS API
```

### Live API

The deployed inference service is available at:

**API**

https://retail-demand-api.lemonrock-804516f2.eastus.azurecontainerapps.io

**Health Check**

https://retail-demand-api.lemonrock-804516f2.eastus.azurecontainerapps.io/health

**Interactive API Documentation**

https://retail-demand-api.lemonrock-804516f2.eastus.azurecontainerapps.io/docs

> The Azure Container App is configured with scale-to-zero capability. The first request after a period of inactivity may therefore experience a short cold-start delay.

### Production Health Check

```http
GET /health
```

Example response from the deployed Azure service:

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

This confirms that the cloud container is running and the serialized XGBoost model has been loaded successfully.

### Cloud Inference

Predictions are generated through:

```http
POST /predict
```

Example request:

```json
{
  "store": 1,
  "item": 1,
  "year": 2017,
  "month": 10,
  "day": 1,
  "day_of_week": 6,
  "week_of_year": 39,
  "quarter": 4,
  "is_weekend": 1,
  "lag_1": 22.0,
  "lag_7": 24.0,
  "lag_14": 21.0,
  "lag_28": 23.0,
  "rolling_mean_7": 23.4,
  "rolling_mean_14": 22.8,
  "rolling_mean_28": 23.1,
  "rolling_std_7": 4.2,
  "rolling_std_28": 5.1
}
```

Example response obtained from the deployed Azure service:

```json
{
  "predicted_sales": 26.25
}
```

This validates the complete inference path from an external HTTPS request to the containerized FastAPI application and trained XGBoost model running on Microsoft Azure.

### Cloud Configuration

The Azure deployment uses:

- **Microsoft Azure Container Apps**
- External HTTPS ingress
- Container image hosted on GHCR
- FastAPI application on port `8000`
- Scale-to-zero configuration
- Minimum replicas: `0`
- Maximum replicas: `1`

The scale-to-zero configuration reduces resource usage when the portfolio API is inactive while keeping the service publicly accessible when needed.

---

## Model Monitoring

Production inference requests are logged using **JSON Lines (JSONL)**.

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

This creates an inference history that can later be combined with observed sales to monitor prediction performance.

The current monitoring implementation focuses on **inference logging and feature-distribution monitoring** rather than a full production observability platform.

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

This provides a simple mechanism for detecting potential differences between the historical training distribution and incoming inference data.

The current implementation is intentionally lightweight. A production extension could incorporate:

- Population Stability Index (PSI)
- Kolmogorov-Smirnov tests
- Evidently
- Automated alerts
- Prediction-performance monitoring
- Automated retraining policies

---

## Interactive Dashboard

A **Streamlit dashboard** provides a business-oriented view of model performance.

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

The dashboard provides a visual layer for exploring model results independently from the REST inference API.

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

### Machine Learning

`Python` · `pandas` · `NumPy` · `scikit-learn` · `XGBoost`

### Data & Analytics

`PostgreSQL` · `SQL` · `Time Series` · `Feature Engineering`

### Backend

`FastAPI` · `Pydantic`

### Cloud & Containerization

`Microsoft Azure` · `Azure Container Apps` · `Docker` · `GitHub Container Registry`

### MLOps & Testing

`Pytest` · `GitHub Actions` · `CI/CD` · `Model Serialization` · `Inference Logging` · `Data Drift Detection`

### Visualization

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

The current API also expects engineered lag and rolling features as input.

In a production architecture, these features would typically be generated automatically by an upstream feature pipeline or retrieved from a feature store rather than manually supplied by an API consumer.

The current model is also limited to **one-step-ahead forecasting** and does not directly generate a multi-day demand trajectory.

---

## Future Improvements

Potential extensions include:

- Automated model retraining
- Automated continuous deployment from GHCR to Azure Container Apps
- Prediction-performance monitoring after actual demand becomes available
- PSI and statistical drift tests
- MLflow experiment tracking and model registry
- Automated feature pipeline or feature store
- Multi-step demand forecasting
- Promotion and price features
- Holiday and event features
- Inventory optimization and reorder-point recommendations
- Deep-learning forecasting experiments
- Automated cloud monitoring and alerting

---

## Reproducibility

### 1. Clone the repository

```bash
git clone https://github.com/0311869uaslp-a11y/retail-demand-forecasting.git
cd retail-demand-forecasting
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run tests

```bash
pytest -v
```

### 6. Run the API locally

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

### 7. Open the API documentation

```text
http://localhost:8000/docs
```

Alternatively, the application can be executed using Docker:

```bash
docker build -t retail-demand-api .
docker run -p 8000:8000 retail-demand-api
```

---

## Author

**José Luis Romero Vázquez**

Electronics Engineer and Data Scientist with international graduate education in Electronic Engineering, Telecommunications, and Computer Networks across Mexico and France, with applied experience in Machine Learning, time-series forecasting, IoT analytics, research, software development, and cloud-based ML deployment.

Professional and research experience across **Mexico, France, and Lithuania**.

**LinkedIn:**  
https://www.linkedin.com/in/jose-luis-romero-vazquez-486569209

**GitHub:**  
https://github.com/0311869uaslp-a11y
