# 🌍 Air Quality Intelligence Platform

> Predicting urban air quality using real sensor data from the UCI Air Quality dataset 
> and live multi-city measurements from the OpenAQ API.

## Project Overview
This end-to-end data science project predicts air quality index (AQI) categories 
and pollutant concentrations using machine learning. It combines a classic benchmark 
dataset with a live REST API to demonstrate big-data ingestion, time-series feature 
engineering, model interpretability (SHAP), and interactive deployment.

## Tech Stack
- **Data**: UCI Air Quality Dataset + OpenAQ REST API v3
- **Processing**: pandas, numpy
- **Modelling**: scikit-learn, XGBoost
- **Explainability**: SHAP
- **Visualization**: matplotlib, seaborn
- **Dashboard**: Streamlit

## Setup
```bash
git clone https://github.com/emaadkalantarii/Urban-Air-Quality-Intelligence-Platform.git
cd air-quality-intelligence
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env            # Then add your OpenAQ API key
python src/data_loader.py
```

## Project Structure
air-quality-intelligence/
├── data/
│   ├── raw/          # Original datasets (not in version control)
│   └── processed/    # Cleaned, feature-engineered data
├── notebooks/        # EDA and experimental notebooks
├── src/              # Production Python scripts
├── outputs/          # Charts, model files
└── requirements.txt

## Phases
- [x] Phase 1 — Environment & Data Acquisition
- [ ] Phase 2 — Exploratory Data Analysis
- [ ] Phase 3 — Feature Engineering
- [ ] Phase 4 — Modelling
- [ ] Phase 5 — Evaluation & SHAP Interpretability
- [ ] Phase 6 — Streamlit Dashboard
- [ ] Phase 7 — Deployment
