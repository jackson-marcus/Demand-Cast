# DemandCast — Production Retail Demand Forecasting Engine

<div align="center">

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![MLflow](https://img.shields.io/badge/MLflow-Registry-0194E2.svg?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Tests: Pytest](https://img.shields.io/badge/tests-pytest-blue.svg?logo=pytest&logoColor=white)](https://pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

> **Hierarchical, multi-horizon demand forecasting engine built on Walmart M5 retail data featuring rolling-origin backtesting, probabilistic LightGBM forecasts, and automated safety stock planning.**

---

## 📖 Executive Summary & Value Proposition

**`demandcast`** is a production-grade, end-to-end machine learning system built with strict engineering discipline, reproducible pipelines, and enterprise MLOps best practices. It bridges the gap between theoretical statistical rigor and high-availability operational microservices.

## 📈 Core Methodologies & Time Series Engineering

### 1. Feature Store & Temporal Signal Processing
- Multi-lag autoregressive features ($t-1, t-7, t-14, t-28, t-365$).
- Rolling window statistics (mean, std, min, max, skew over 7, 14, 28, and 90-day windows).
- Calendar, seasonality, SNAP food assistance event flags, and price elasticity ratios.

### 2. Rolling-Origin Backtesting & Benchmark Suite
- Expanding-window evaluation simulating realistic production retraining schedules without data leakage.
- Direct comparison against classical statistical baselines (AutoARIMA, Exponential Smoothing, Seasonal Naive) showing >20% WAPE reduction.

### 3. Probabilistic Forecasts & Safety Stock Calculus
- P10, P50, and P90 quantile loss optimization.
- Computes dynamic safety stock and reorder points based on forecast uncertainty and targeted service levels (e.g. 95% or 99% fill rate):
$$	ext{Safety Stock} = z_lpha \cdot \sqrt{L \cdot \sigma_D^2 + D^2 \cdot \sigma_L^2}$$

## 📊 Architecture & Pipeline

```mermaid
flowchart LR
    Raw[M5 Walmart Dataset] --> Feat[Lag & Rolling Feature Store]
    Feat --> Backtest[Rolling-Origin Backtesting Engine]
    Backtest --> LGBM[Probabilistic LightGBM<br/>P10, P50, P90]
    LGBM --> Stock[Safety Stock & Reorder Planner]
    Stock --> API[FastAPI :8060] --> UI[Streamlit Demand Dashboard :8561]
```

## 🛠️ Tech Stack & Engineering Standards
- **Forecasting:** Python 3.12, StatsForecast, MLForecast, LightGBM, UtilsForecast, PyArrow
- **Serving & UI:** FastAPI, Streamlit, MLflow
- **Testing:** Pytest coverage across feature generation, backtesting loops, and inventory metrics


---

## 🚀 Quickstart & Setup Guide

### 1. Prerequisites & Environment Setup
Using **[uv](https://docs.astral.sh/uv/)** for lightning-fast, reproducible dependency resolution:

```bash
# Clone the repository
git clone https://github.com/jackson-marcus/demandcast.git
cd demandcast

# Install dependencies and pre-commit hooks
uv sync --group dev
```

### 2. Run Test Suite & Code Quality Checks
```bash
# Run unit & integration tests with coverage
uv run pytest --cov

# Run ruff linter and formatting checks
uv run ruff check .
uv run ruff format --check .
```

### 3. Launch Services Locally
```bash
# Start FastAPI REST API (listening on port :8060)
make api
# Or: uv run uvicorn demandcast.api.main:app --reload --port 8060

# Start interactive Streamlit dashboard (listening on port :8561)
make ui

# Launch local MLflow Experiment Tracking UI (listening on port :5006)
make mlflow
```

### 4. Run with Docker Compose
```bash
# Spin up the complete microservice stack
docker compose up --build
```

---

## 📂 Repository Layout

```
demandcast/
├── .github/workflows/ci.yml       # GitHub Actions CI pipeline (lint, test, build)
├── configs/                      # Configuration files and hyperparameters
├── data/                         # Data directory (raw, interim, processed)
├── scripts/                      # Data generators and operational scripts
├── src/demandcast/               # Core Python package
│   ├── api/                      # FastAPI routes, schemas, and endpoints
│   ├── models/                   # Statistical models, ML algorithms, and estimators
│   ├── ui/                       # Streamlit interactive application
│   └── settings.py               # Centralized configuration & environment loader
├── tests/                        # Comprehensive Pytest suite
├── docker-compose.yml            # Multi-service container orchestration
├── Dockerfile                    # Container definition for API service
├── Makefile                      # Standardized project tasks
└── pyproject.toml                # Pinned dependencies and tool configs
```

---

## 👤 Author & Contact

**Jackson Marcus**
- **Email:** [jackson.marcus.work@gmail.com](mailto:jackson.marcus.work@gmail.com)
- **Upwork:** [Jackson Marcus on Upwork](https://www.upwork.com/freelancers/~012235717501ad9c7b)
- **GitHub:** [@jackson-marcus](https://github.com/jackson-marcus)

*Available for machine learning engineering, MLOps, data science, and AI system architecture consulting and contract engagements.*

