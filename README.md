# Urban Air Quality Intelligence Platform

> An end-to-end machine learning system that predicts urban air quality using 9,357 hourly sensor readings from the UCI Air Quality dataset combined with live multi-city measurements from the OpenAQ API v3 — deployed as an interactive web application.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://YOUR_APP_URL.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Live Demo

**[→ Open the interactive dashboard](https://YOUR_APP_URL.streamlit.app)**

The dashboard has four pages:
- **Overview** — project summary, AQI category legend, dataset metrics, EDA findings
- **Live Predictor** — adjust sensor sliders and get an instant XGBoost AQI prediction with confidence breakdown
- **SHAP Explainer** — global feature importance, beeswarm plot, and per-prediction waterfall charts
- **OpenAQ Live Data** — fetch real-time NO₂ measurements across Italy, France, and Germany via the OpenAQ API

---

## Project Overview

Air quality prediction is a genuine public health challenge. This project builds a production-grade ML pipeline that:

1. Ingests data from two sources: a historic Italian sensor dataset (UCI) and a live REST API (OpenAQ v3)
2. Engineers 46 features including **lag features**, **rolling window statistics**, and **cyclical time encodings** — concepts directly applicable to streaming/big data pipelines
3. Trains and compares classification and regression models with proper time-aware train/validation/test splitting
4. Explains every model prediction using **SHAP** (SHapley Additive exPlanations)
5. Deploys the complete system as a multi-page interactive Streamlit application

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| Data manipulation | Pandas · NumPy |
| Machine learning | Scikit-learn · XGBoost |
| Model interpretability | SHAP |
| Visualisation | Matplotlib · Seaborn |
| Dashboard | Streamlit |
| Live data | OpenAQ REST API v3 |
| Version control | Git · GitHub |

---

## Dataset

**UCI Air Quality Dataset** — 9,357 hourly readings from a gas multisensor device deployed in an Italian city (March 2004 – April 2005). 13 features including CO, NOx, NO₂, Benzene, O₃, temperature, and humidity.

> Download: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/360/air+quality)  
> Place the file at `data/raw/AirQualityUCI.xlsx` before running Phase 1.

**OpenAQ API v3** — Live multi-city air quality data fetched programmatically from monitoring stations across Italy, France, and Germany. Requires a free API key from [explore.openaq.org/register](https://explore.openaq.org/register).

---

## Results

### Classification — Predicting AQI Category (Good / Moderate / Poor / Very Poor)

| Model | Accuracy | F1 (macro) | F1 (weighted) |
|---|---|---|---|
| Dummy baseline | 65.9% | 0.20 | 0.52 |
| Random Forest | 80.8% | 0.63 | 0.75 |
| **XGBoost** | **86.6%** | **0.74** | **0.84** |

XGBoost outperforms the majority-class baseline by **+20.7 percentage points** in accuracy.

### Regression — Predicting NO₂ Concentration (µg/m³)

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Dummy baseline | 70.39 | 57.60 | -1.094 |
| Random Forest | 17.18 | 10.02 | 0.875 |
| **XGBoost** | **16.49** | **9.37** | **0.885** |

XGBoost explains **88.5% of NO₂ variance** (R² = 0.885) and reduces RMSE from 70.4 to 16.5 µg/m³.

### Top SHAP Features (predicting "Poor" AQI)

| Rank | Feature | Mean \|SHAP\| | Interpretation |
|---|---|---|---|
| 1 | `NO2GT_rolling_3h_mean` | 1.511 | Sustained recent pollution is the strongest signal |
| 2 | `NO2GT_lag_1h` | 0.548 | What happened 1 hour ago — temporal autocorrelation |
| 3 | `NOx(GT)` | 0.505 | NOx is chemically linked to NO₂ production |
| 4 | `NO2GT_change_1h` | 0.484 | Rising trend is a stronger alarm than a stable level |
| 5 | `NO2GT_change_3h` | 0.401 | Longer-window trend confirms the pattern |

Lag and rolling features dominate — the model correctly learned that air quality evolves gradually over time, not randomly from hour to hour.

---

## Project Structure

```
urban-air-quality-intelligence-platform/
│
├── notebooks/
│   ├── 01_data_acquisition.ipynb      # UCI loading + OpenAQ API data fetch
│   ├── 02_eda.ipynb                   # Distributions, time patterns, correlations
│   ├── 03_feature_engineering.ipynb   # Lag/rolling features, cyclical encoding, split
│   ├── 04_modelling.ipynb             # Baseline, Random Forest, XGBoost (2 tracks)
│   └── 05_shap_interpretability.ipynb # Global + local SHAP explanations
│
├── data/
│   ├── raw/            # Original files (gitignored — see setup below)
│   └── processed/      # Cleaned features, scaler, train/val/test splits
│
├── models/
│   ├── xgb_classifier.pkl             # Best classification model
│   └── xgb_regressor.pkl              # Best regression model
│
├── outputs/                           # All charts generated by notebooks (PNG)
│
├── app.py                             # Streamlit dashboard (4 pages)
├── requirements.txt                   # Deployment-safe dependencies
├── .streamlit/
│   └── config.toml                    # App theme and server settings
├── .env.example                       # API key template
└── README.md
```

---

## Setup & Run Locally

### 1. Clone and create environment

```bash
git clone https://github.com/emaadkalantarii/Urban-Air-Quality-Intelligence-Platform.git
cd urban-air-quality-intelligence-platform
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configure API key

```bash
cp .env.example .env
# Open .env and add your OpenAQ API key:
# OPENAQ_API_KEY=your_key_here
```

Get a free key at [explore.openaq.org/register](https://explore.openaq.org/register).

### 3. Download the UCI dataset

Download `AirQualityUCI.xlsx` from the [UCI repository](https://archive.ics.uci.edu/dataset/360/air+quality) and place it at:

```
data/raw/AirQualityUCI.xlsx
```

### 4. Run the notebooks in order

Open VSCode or Jupyter and run each notebook from top to bottom:

```
notebooks/01_data_acquisition.ipynb
notebooks/02_eda.ipynb
notebooks/03_feature_engineering.ipynb
notebooks/04_modelling.ipynb
notebooks/05_shap_interpretability.ipynb
```

Each notebook saves its outputs to `data/processed/`, `models/`, and `outputs/` automatically.

### 5. Launch the dashboard

```bash
streamlit run app.py
```

The app opens automatically at `http://localhost:8501`.

---

## Key Concepts Demonstrated

**Time-series feature engineering** — Lag features (`NO2_lag_1h`, `NO2_lag_24h`) and rolling window statistics (`NO2_rolling_3h_mean`, `NO2_rolling_24h_std`) encode temporal context directly into the feature matrix. This mirrors the sliding-window approach used in real-time streaming pipelines (Kafka, Spark Streaming).

**Cyclical encoding** — Hour, day-of-week, and month are encoded as `(sin, cos)` pairs rather than raw integers. This ensures the model correctly understands that hour 23 and hour 0 are adjacent (1 hour apart), not 23 units apart.

**Time-aware splitting** — The dataset is split chronologically (70% train / 15% val / 15% test) rather than randomly. Random splitting on time-series data causes data leakage — the model would train on the future of its own test set.

**Class imbalance handling** — The AQI distribution is heavily skewed toward "Poor" (57.5%) and "Moderate" (34.6%). `class_weight="balanced"` in sklearn and `sample_weight` in XGBoost automatically penalise misclassification of rare classes (Good: 3.7%, Very Poor: 4.2%).

**SHAP interpretability** — `shap.TreeExplainer` computes exact (not approximate) Shapley values for XGBoost, producing globally interpretable feature importance rankings and locally interpretable per-prediction waterfall charts.

**Live API integration** — The OpenAQ v3 REST API is called programmatically at both data acquisition time (notebooks) and runtime (dashboard), demonstrating a real data pipeline rather than a static dataset workflow.

---

## Notebooks on GitHub

Each notebook renders completely on GitHub — every cell output, chart, and table is visible without running any code. This makes the analytical workflow fully transparent to any reviewer.

| Notebook | Key output |
|---|---|
| [01_data_acquisition](notebooks/01_data_acquisition.ipynb) | 9,357 UCI rows loaded · 843 OpenAQ NO₂ measurements fetched |
| [02_eda](notebooks/02_eda.ipynb) | Rush-hour peaks · seasonal patterns · correlation matrix |
| [03_feature_engineering](notebooks/03_feature_engineering.ipynb) | 46 features · time-aware 70/15/15 split · cyclical encoding explainer |
| [04_modelling](notebooks/04_modelling.ipynb) | XGBoost 86.6% accuracy · R²=0.885 · confusion matrices |
| [05_shap_interpretability](notebooks/05_shap_interpretability.ipynb) | Beeswarm · waterfall · dependence plots |

---

## License

MIT — free to use, adapt, and reference with attribution.
