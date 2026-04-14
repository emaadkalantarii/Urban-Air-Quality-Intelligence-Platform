# ============================================================
# app.py  —  Air Quality Intelligence Platform
# Streamlit dashboard for the end-to-end DS portfolio project
#
# Run locally:  streamlit run app.py
# Deploy:       Push to GitHub → connect on share.streamlit.io
# ============================================================

# ── Imports ──────────────────────────────────────────────────────
import os
import json
import warnings
import time
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import requests
import joblib
import streamlit as st

warnings.filterwarnings("ignore")

# ── Page configuration ────────────────────────────────────────────
# Must be the FIRST Streamlit call in the script — before any other st.*
st.set_page_config(
    page_title="Air Quality Intelligence",
    page_icon="🌍",
    layout="wide",               # Use full browser width
    initial_sidebar_state="expanded"
)

# ── Constants ─────────────────────────────────────────────────────
# NOTE: CSS injection (st.markdown) must NOT be at module level.
# It is injected inside main() at runtime instead.
AQI_LABELS   = ["Good", "Moderate", "Poor", "Very Poor"]
AQI_COLORS   = ["#4caf50", "#ff9800", "#f44336", "#9c27b0"]
AQI_ADVISORIES = {
    0: "Air quality is satisfactory. No health risk.",
    1: "Acceptable quality. Sensitive groups should limit prolonged outdoor exertion.",
    2: "Health effects possible for the general public. Reduce outdoor activities.",
    3: "Serious health effects. Avoid outdoor activities. Wear a mask if going outside.",
}
AQI_ICONS = {0: "✅", 1: "⚠️", 2: "🔴", 3: "🟣"}

# ── Cached resource loaders ───────────────────────────────────────
# @st.cache_resource: runs ONCE and caches the result forever.
# On every subsequent rerun (every slider move), returns the cached object.
# Perfect for large objects like ML models that are expensive to load.

@st.cache_resource
def load_models():
    """Load the trained XGBoost classifier and regressor from disk."""
    cls_path = "models/xgb_classifier.pkl"
    reg_path = "models/xgb_regressor.pkl"

    if not os.path.exists(cls_path) or not os.path.exists(reg_path):
        return None, None

    clf = joblib.load(cls_path)
    reg = joblib.load(reg_path)
    return clf, reg


@st.cache_resource
def load_scaler():
    """Load the fitted StandardScaler from Phase 3."""
    path = "data/processed/scaler.pkl"
    if not os.path.exists(path):
        return None
    return joblib.load(path)


@st.cache_data
def load_feature_cols():
    """Load the ordered list of feature column names from Phase 3."""
    path = "data/processed/feature_cols.json"
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_test_data():
    """Load the processed test split from Phase 3."""
    path = "data/processed/test.csv"
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
    return df


# ── Feature vector builder ────────────────────────────────────────

def build_feature_vector(inputs: dict, feature_cols: list) -> pd.DataFrame:
    """
    Constructs a single-row DataFrame matching the exact feature schema
    expected by the model, from a dictionary of user-provided inputs.

    Any feature not explicitly provided is filled with 0.0 as a safe default.
    In practice, the most important features (lag, rolling, sensor values)
    are all provided via the sliders.

    Args:
        inputs:       dict of {feature_name: value} from the UI sliders
        feature_cols: ordered list of all feature column names

    Returns:
        Single-row pd.DataFrame with all feature columns
    """
    row = {col: inputs.get(col, 0.0) for col in feature_cols}
    # Dict comprehension: for each expected column, use the provided value
    # or fall back to 0.0 if the user didn't set it.
    return pd.DataFrame([row])


def cyclical_encode(value: float, period: float):
    """
    Encodes a cyclical value (hour, day, month) as (sin, cos) pair.
    See Phase 3 notebook for the full explanation.
    """
    sin_val = np.sin(2 * np.pi * value / period)
    cos_val = np.cos(2 * np.pi * value / period)
    return sin_val, cos_val


# ── Sidebar ───────────────────────────────────────────────────────

def render_sidebar():
    """Renders the sidebar navigation and project info."""
    with st.sidebar:
        st.image(
            "https://img.shields.io/badge/Air%20Quality-Intelligence-blue?style=for-the-badge",
            use_container_width=True
        )
        st.markdown("---")

        # Navigation radio buttons
        # st.radio() returns the currently selected option as a string.
        page = st.radio(
            "Navigate",
            ["🏠 Overview",
             "🔮 Live Predictor",
             "🧠 SHAP Explainer",
             "🌐 OpenAQ Live Data"],
            label_visibility="collapsed"
            # label_visibility="collapsed" hides the "Navigate" label
            # but keeps it accessible for screen readers
        )

        st.markdown("---")
        st.markdown("### Project info")
        st.markdown("""
        **Dataset:** UCI Air Quality (9,357 hourly readings, Italy 2004–2005)

        **Live data:** OpenAQ API v3 (IT · FR · DE)

        **Models:** XGBoost classifier + regressor

        **Features:** 46 engineered features including lag, rolling, and cyclical time encodings
        """)

        st.markdown("---")

        # Model status indicator
        clf, reg = load_models()
        if clf is not None:
            st.success("Models loaded ✓")
        else:
            st.error("Models not found — run Phase 4 first")

        st.markdown("---")
        st.markdown("""
        **Built by:**Emad Kalantari
        """)
        st.markdown(
            "[![GitHub](https://img.shields.io/badge/Source-GitHub-black?logo=github)]"
            "(https://github.com/emaadkalantarii/Urban-Air-Quality-Intelligence-Platform.git)"
        )
        # ↑ Replace YOUR_USERNAME with your actual GitHub username

    return page


# ── PAGE 1: Overview ──────────────────────────────────────────────

def page_overview():
    st.title("🌍 Air Quality Intelligence Platform")
    st.markdown(
        "An end-to-end machine learning system that predicts urban air quality "
        "using historic sensor data and live multi-city measurements from the OpenAQ API."
    )

    # ── AQI category legend ───────────────────────────────────────
    st.markdown("### Air Quality Index (AQI) categories")
    cols = st.columns(4)
    # st.columns(n) creates n side-by-side columns.
    # Each column is a context manager you write into.
    for col, label, color, advisory in zip(
        cols, AQI_LABELS, AQI_COLORS, AQI_ADVISORIES.values()
    ):
        with col:
            st.markdown(
                f"""<div style="background:{color}22; border-left:4px solid {color};
                padding:12px; border-radius:4px; margin-bottom:8px; min-height:90px;">
                <b style="color:{color}">{label}</b><br>
                <small>{advisory}</small></div>""",
                unsafe_allow_html=True
                # unsafe_allow_html=True lets us inject raw HTML.
                # Use sparingly — only for styling that Streamlit's
                # built-in components can't achieve.
            )

    st.markdown("---")

    # ── Dataset summary metrics ───────────────────────────────────
    st.markdown("### Dataset at a glance")
    df_test = load_test_data()

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("UCI hourly readings", "9,357",
                  help="Hourly sensor readings from an Italian city, Jan 2004 – Dec 2005")
    with m2:
        st.metric("Engineered features", "46",
                  help="Including lag features, rolling windows, cyclical time encoding")
    with m3:
        st.metric("Countries (OpenAQ)", "3",
                  help="Italy · France · Germany — live data via OpenAQ API v3")
    with m4:
        st.metric("Target variable", "AQI category",
                  help="4-class classification: Good / Moderate / Poor / Very Poor")
    # st.metric() displays a labelled value with an optional delta indicator.
    # The help= parameter shows a tooltip on hover.

    st.markdown("---")

    # ── Model performance summary ─────────────────────────────────
    st.markdown("### Model performance (test set)")

    # Classification results (test set) — from Phase 4 notebook
    cls_data = {
        "Model":      ["Dummy baseline", "Random Forest", "XGBoost"],
        "Accuracy":   ["65.9%", "80.8%", "86.6%"],
        "F1 (macro)": ["0.20",  "0.63",  "0.74"],
        "F1 (weighted)": ["0.52", "0.75", "0.84"],
    }
    # Regression results (test set) — from Phase 4 notebook
    reg_data = {
        "Model":  ["Dummy baseline", "Random Forest", "XGBoost"],
        "RMSE (µg/m³)": ["70.39", "17.18", "16.49"],
        "MAE (µg/m³)":  ["57.60", "10.02",  "9.37"],
        "R²":           ["-1.094", "0.875", "0.885"],
    }

    cls_col, reg_col = st.columns(2)
    with cls_col:
        st.markdown("**Classification — predict AQI category**")
        st.dataframe(pd.DataFrame(cls_data), use_container_width=True, hide_index=True)
    with reg_col:
        st.markdown("**Regression — predict NO₂ concentration**")
        st.dataframe(pd.DataFrame(reg_data), use_container_width=True, hide_index=True)

    st.caption(
        "XGBoost outperforms the dummy baseline by +20.7 percentage points in accuracy "
        "and reduces RMSE from 70.4 to 16.5 µg/m³ — explaining 88.5% of NO₂ variance (R²=0.885)."
    )

    st.info(
        "💡 **How to read this project:** "
        "Use the **Live Predictor** page to get a real-time AQI prediction "
        "by adjusting sensor readings and time. Use the **SHAP Explainer** "
        "to understand *why* the model makes each prediction."
    )

    st.markdown("---")

    # ── EDA charts from Phase 2 ───────────────────────────────────
    st.markdown("### Key findings from exploratory data analysis")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        if os.path.exists("outputs/02_hourly_patterns.png"):
            st.image(
                "outputs/02_hourly_patterns.png",
                caption="Pollution peaks during morning rush hour (7–9am)",
                use_container_width=True
            )
        else:
            st.warning("Run Phase 2 notebook to generate EDA charts")

    with chart_col2:
        if os.path.exists("outputs/02_correlation_heatmap.png"):
            st.image(
                "outputs/02_correlation_heatmap.png",
                caption="Feature correlation matrix — sensor proxies vs actual concentrations",
                use_container_width=True
            )


# ── PAGE 2: Live Predictor ────────────────────────────────────────

def page_predictor():
    st.title("🔮 Live AQI Predictor")
    st.markdown(
        "Adjust the sensor readings and time parameters below. "
        "The XGBoost model predicts the AQI category and NO₂ concentration **instantly**."
    )

    clf, reg   = load_models()
    scaler     = load_scaler()
    feat_cols  = load_feature_cols()

    if clf is None or scaler is None or not feat_cols:
        st.error(
            "Model files not found. Make sure you have run Phases 3 and 4 "
            "and that `models/` and `data/processed/` are committed to your repo."
        )
        return

    # ── Input sliders ─────────────────────────────────────────────
    st.markdown("### Sensor & weather inputs")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("**Pollutant sensors**")
        no2_lag1   = st.slider("NO₂ (1h ago) µg/m³",    0.0, 300.0, 80.0,  step=1.0,
                                help="NO₂ reading from 1 hour ago — the single strongest predictor")
        no2_lag24  = st.slider("NO₂ (24h ago) µg/m³",   0.0, 300.0, 70.0,  step=1.0)
        co_gt      = st.slider("CO concentration mg/m³", 0.0,  10.0,  2.0,  step=0.1)
        nox_gt     = st.slider("NOx concentration µg/m³",0.0, 500.0,100.0,  step=5.0)

    with col_b:
        st.markdown("**Derived / rolling features**")
        no2_roll3  = st.slider("NO₂ rolling 3h mean",    0.0, 300.0, 85.0, step=1.0,
                                help="Average NO₂ over the last 3 hours — top SHAP feature")
        no2_roll24 = st.slider("NO₂ rolling 24h mean",   0.0, 300.0, 75.0, step=1.0)
        no2_chg1   = st.slider("NO₂ change (last 1h)",  -100.0, 100.0, 5.0, step=1.0,
                                help="Positive = rising pollution, negative = falling")
        pt08_s1    = st.slider("PT08.S1 sensor (CO)",   600.0, 2000.0, 1100.0, step=10.0)

    with col_c:
        st.markdown("**Time & weather**")
        hour       = st.slider("Hour of day",      0, 23, 8,
                                help="8am = morning rush hour → typically higher pollution")
        dow        = st.slider("Day of week",      0, 6,  1,
                                help="0=Monday ... 6=Sunday")
        month      = st.slider("Month",            1, 12, 6)
        temp       = st.slider("Temperature (°C)", -5.0, 40.0, 18.0, step=0.5)
        humidity   = st.slider("Relative humidity (%)", 10.0, 100.0, 55.0, step=1.0)

    # ── Build feature vector ──────────────────────────────────────
    # Compute cyclical encodings from raw time values
    h_sin, h_cos   = cyclical_encode(hour,  24)
    d_sin, d_cos   = cyclical_encode(dow,   7)
    m_sin, m_cos   = cyclical_encode(month, 12)
    is_weekend     = int(dow >= 5)

    # Map slider values to the exact feature names from feature_cols.json
    inputs = {
        # Pollutant ground-truth readings
        "CO(GT)":          co_gt,
        "NOx(GT)":         nox_gt,
        "C6H6(GT)":        co_gt * 3.5,   # Approximate relationship
        # Sensor proxies
        "PT08.S1(CO)":     pt08_s1,
        "PT08.S2(NMHC)":   900.0,
        "PT08.S3(NOx)":    1100.0 - nox_gt * 0.8,
        "PT08.S4(NO2)":    1200.0 + no2_lag1 * 2,
        "PT08.S5(O3)":     900.0,
        # Weather
        "T":    temp,
        "RH":   humidity,
        "AH":   (humidity / 100) * 0.02 * (temp + 273.15) / 10,
        # Time features (raw)
        "Hour":      hour,
        "DayOfWeek": dow,
        "Month":     month,
        "IsWeekend": is_weekend,
        # Cyclical time encodings
        "Hour_sin":  h_sin,  "Hour_cos":  h_cos,
        "DoW_sin":   d_sin,  "DoW_cos":   d_cos,
        "Month_sin": m_sin,  "Month_cos": m_cos,
        # Lag features
        "NO2GT_lag_1h":   no2_lag1,
        "NO2GT_lag_2h":   no2_lag1 * 0.97,
        "NO2GT_lag_3h":   no2_lag1 * 0.95,
        "NO2GT_lag_6h":   no2_lag1 * 0.90,
        "NO2GT_lag_12h":  no2_lag1 * 0.85,
        "NO2GT_lag_24h":  no2_lag24,
        "COGT_lag_1h":    co_gt * 0.98,
        "COGT_lag_3h":    co_gt * 0.93,
        "COGT_lag_24h":   co_gt * 0.88,
        "NOxGT_lag_1h":   nox_gt * 0.97,
        "NOxGT_lag_24h":  nox_gt * 0.88,
        "T_lag_1h":       temp - 0.3,
        "T_lag_24h":      temp - 1.5,
        "RH_lag_1h":      humidity + 0.5,
        "RH_lag_24h":     humidity + 2.0,
        # Rolling features
        "NO2GT_rolling_3h_mean":  no2_roll3,
        "NO2GT_rolling_6h_mean":  (no2_roll3 + no2_roll24) / 2,
        "NO2GT_rolling_24h_mean": no2_roll24,
        "NO2GT_rolling_24h_std":  abs(no2_chg1) * 2.5,
        "COGT_rolling_3h_mean":   co_gt * 0.99,
        "COGT_rolling_24h_mean":  co_gt * 0.95,
        "T_rolling_3h_mean":      temp,
        "T_rolling_24h_mean":     temp - 0.5,
        # Rate of change
        "NO2GT_change_1h": no2_chg1,
        "NO2GT_change_3h": no2_chg1 * 2.8,
    }

    X_input = build_feature_vector(inputs, feat_cols)
    # Scale using the SAME scaler fitted in Phase 3
    X_scaled = scaler.transform(X_input)

    # ── Run predictions ───────────────────────────────────────────
    aqi_pred       = int(clf.predict(X_scaled)[0])
    # .predict() returns an array — [0] gets the single prediction
    aqi_proba      = clf.predict_proba(X_scaled)[0]
    # .predict_proba() returns probabilities for each class.
    # Shape: (1, 4) → [0] gives the 4 probabilities for this one sample.
    no2_pred       = float(reg.predict(X_scaled)[0])

    # ── Display results ───────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Prediction results")

    res_col1, res_col2 = st.columns([1, 2])

    with res_col1:
        color = AQI_COLORS[aqi_pred]
        icon  = AQI_ICONS[aqi_pred]
        st.markdown(
            f"""<div style="background:{color}22; border:2px solid {color};
            border-radius:12px; padding:24px; text-align:center;">
            <div style="font-size:48px">{icon}</div>
            <div style="font-size:28px; font-weight:bold; color:{color}">
                {AQI_LABELS[aqi_pred]}</div>
            <div style="font-size:14px; color:#666; margin-top:8px">
                AQI Category Prediction</div>
            <hr style="border-color:{color}44">
            <div style="font-size:22px; font-weight:bold">
                NO₂ ≈ {no2_pred:.1f} µg/m³</div>
            <div style="font-size:12px; color:#666">Regression forecast</div>
            </div>""",
            unsafe_allow_html=True
        )

        st.markdown(f"""
        <div style="background:#f8f9fa; border-radius:8px; padding:12px; margin-top:12px;
        font-size:13px; line-height:1.6;">
        {AQI_ADVISORIES[aqi_pred]}
        </div>""", unsafe_allow_html=True)

    with res_col2:
        # Probability bar chart for all 4 classes
        st.markdown("**Confidence per AQI class**")

        fig, ax = plt.subplots(figsize=(7, 3))
        bars = ax.barh(
            AQI_LABELS,
            aqi_proba,
            color=AQI_COLORS,
            edgecolor="white",
            linewidth=0.5,
            height=0.55
        )
        # Highlight the predicted class with a thicker border
        bars[aqi_pred].set_edgecolor("black")
        bars[aqi_pred].set_linewidth(2)

        for bar, prob in zip(bars, aqi_proba):
            ax.text(
                bar.get_width() + 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{prob:.1%}",    # Format as percentage: 0.823 → "82.3%"
                va="center", fontsize=10
            )

        ax.set_xlim(0, 1.15)
        ax.set_xlabel("Predicted probability")
        ax.set_title("Model confidence by AQI class", fontsize=11)
        ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)
        # st.pyplot(fig) renders a matplotlib figure in Streamlit.
        # Always pass the figure object explicitly — don't rely on plt.show().
        plt.close(fig)
        # plt.close() frees memory — important in Streamlit which reruns often.

    # ── Context note ──────────────────────────────────────────────
    st.caption(
        f"Prediction made at {datetime.now().strftime('%H:%M:%S')} · "
        f"Model: XGBoost · Features: {len(feat_cols)} · "
        f"Training data: UCI Air Quality 2004–2005"
    )


# ── PAGE 3: SHAP Explainer ────────────────────────────────────────

def page_shap():
    st.title("🧠 SHAP Model Explainer")
    st.markdown(
        "SHAP (SHapley Additive exPlanations) reveals **why** the model makes each prediction. "
        "This section shows which features drive AQI predictions globally and locally."
    )

    st.info(
        "**How to read SHAP values:** A positive SHAP value means a feature "
        "*increased* the model's prediction toward a worse AQI category. "
        "A negative value means it *decreased* it (pushed toward Good)."
    )

    tab1, tab2, tab3 = st.tabs(
        ["Global importance", "Beeswarm plot", "Waterfall — local explanation"]
    )
    # st.tabs() creates clickable tabs — each is a context manager.
    # Content inside each tab is only rendered when that tab is active.

    with tab1:
        st.markdown("#### Which features matter most globally?")
        st.markdown(
            "The bar chart shows mean absolute SHAP value per feature, "
            "averaged across all test predictions. "
            "Longer bar = more influential feature overall."
        )
        if os.path.exists("outputs/05_shap_global_importance.png"):
            st.image("outputs/05_shap_global_importance.png", use_container_width=True)
        else:
            st.warning("Run Phase 5 notebook to generate SHAP charts.")

        st.markdown("""
        **Key insight:** The top features are consistently the **lag and rolling features**
        engineered in Phase 3 — confirming that recent pollution history is the strongest
        predictor of current air quality. This validates both the feature engineering
        decisions and the physical reality that air pollution evolves gradually.
        """)

    with tab2:
        st.markdown("#### Beeswarm — every prediction, every feature")
        st.markdown(
            "Each dot is one test prediction. "
            "The x-axis shows the SHAP value. "
            "**Red dots** = high feature value. **Blue dots** = low feature value."
        )
        if os.path.exists("outputs/05_shap_beeswarm.png"):
            st.image("outputs/05_shap_beeswarm.png", use_container_width=True)
        else:
            st.warning("Run Phase 5 notebook to generate SHAP charts.")

        st.markdown("""
        **How to read this:** If `NO2GT_rolling_3h_mean` shows red dots on the right
        (positive SHAP), it means high recent NO₂ consistently pushes the model toward
        predicting worse air quality — exactly what we would expect physically.
        """)

    with tab3:
        st.markdown("#### Waterfall — why did the model predict THIS specific hour?")

        wf_option = st.selectbox(
            "Choose a case to explain:",
            ["Correctly predicted Poor air quality",
             "Correctly predicted Good air quality",
             "Misclassified case (model made an error)"]
        )
        # st.selectbox() renders a dropdown menu.
        # Returns the currently selected string.

        wf_files = {
            "Correctly predicted Poor air quality":  "outputs/05_waterfall_correct_poor.png",
            "Correctly predicted Good air quality":  "outputs/05_waterfall_correct_good.png",
            "Misclassified case (model made an error)": "outputs/05_waterfall_misclassified.png",
        }
        wf_path = wf_files[wf_option]

        if os.path.exists(wf_path):
            st.image(wf_path, use_container_width=True)
        else:
            st.warning("Run Phase 5 notebook to generate waterfall plots.")

        st.markdown("""
        **How to read this:** The chart reads bottom-to-top.
        Starting from the base value (average model prediction),
        each bar adds or subtracts based on that feature's value for this specific hour.
        The final sum is the model's actual output for this prediction.
        Red bars push toward "Poor", blue bars push toward "Good".
        """)

    # Regression SHAP
    st.markdown("---")
    st.markdown("#### SHAP for the regression model (NO₂ concentration)")
    if os.path.exists("outputs/05_shap_regression_importance.png"):
        st.image("outputs/05_shap_regression_importance.png", use_container_width=True)
    st.markdown(
        "The regression SHAP values are in **µg/m³** — directly interpretable. "
        "A SHAP value of +33 for `NO2GT_rolling_3h_mean` means: when the 3-hour "
        "rolling mean is high, the model's NO₂ forecast increases by ~33 µg/m³ "
        "compared to its average prediction."
    )


# ── PAGE 4: OpenAQ Live Data ──────────────────────────────────────

def page_openaq():
    st.title("🌐 OpenAQ Live Air Quality Data")
    st.markdown(
        "Fetch real-time NO₂ measurements from the "
        "[OpenAQ API v3](https://docs.openaq.org) across Italy, France, and Germany. "
        "This demonstrates the **big data ingestion** component of the project — "
        "a live API pipeline integrated directly into the dashboard."
    )

    # API key from environment variable — never hardcode keys in source
    try:
        api_key = st.secrets["OPENAQ_API_KEY"]
    except (KeyError, FileNotFoundError):
        api_key = os.environ.get("OPENAQ_API_KEY", "")
    # st.secrets reads from .streamlit/secrets.toml locally,
    # and from the Streamlit Cloud secrets manager in production.
    # We fall back to environment variable if secrets aren't configured.
    # os.environ.get("KEY", default) reads an environment variable.
    # If not set, returns the default (empty string here).
    # On Streamlit Cloud, set this in the app's Secrets manager.

    if not api_key:
        st.warning(
            "OpenAQ API key not found. "
            "Set the `OPENAQ_API_KEY` environment variable, or on Streamlit Cloud "
            "add it under **Settings → Secrets** as `OPENAQ_API_KEY = 'your_key'`."
        )
        # Still show cached data if available
        cached_path = "data/raw/openaq_no2_measurements.csv"
        if os.path.exists(cached_path):
            st.info("Showing cached data from Phase 1 data acquisition.")
            df_cached = pd.read_csv(cached_path)
            df_cached["datetime_utc"] = pd.to_datetime(
                df_cached["datetime_utc"], utc=True, errors="coerce"
            )
            _render_openaq_charts(df_cached)
        return

    col_fetch, col_info = st.columns([1, 3])
    with col_fetch:
        fetch_clicked = st.button(
            "Fetch latest data",
            type="primary",
            help="Calls the OpenAQ API and retrieves the last 7 days of NO₂ data"
        )
        # st.button() returns True on the rerun immediately after being clicked,
        # False on all other reruns. This is Streamlit's event model.

    with col_info:
        st.caption(
            "Fetches from IT · FR · DE monitoring stations · "
            "Uses OpenAQ API v3 · Rate limited: please don't click repeatedly"
        )

    if fetch_clicked:
        with st.spinner("Fetching live data from OpenAQ API..."):
            # st.spinner() shows a loading animation while the block runs.
            df_live = _fetch_openaq_live(api_key)

        if df_live is not None and len(df_live) > 0:
            st.success(f"Fetched {len(df_live):,} measurements ✓")
            _render_openaq_charts(df_live)
        else:
            st.error("No data returned — check your API key or try again later.")
            # Fall back to cached
            cached_path = "data/raw/openaq_no2_measurements.csv"
            if os.path.exists(cached_path):
                st.info("Showing cached data from Phase 1 instead.")
                df_cached = pd.read_csv(cached_path)
                _render_openaq_charts(df_cached)
    else:
        # On first load (before button clicked), show cached data
        cached_path = "data/raw/openaq_no2_measurements.csv"
        if os.path.exists(cached_path):
            st.info(
                "Showing cached data from Phase 1. "
                "Click **Fetch latest data** to get current readings."
            )
            df_cached = pd.read_csv(cached_path)
            df_cached["datetime_utc"] = pd.to_datetime(
                df_cached["datetime_utc"], utc=True, errors="coerce"
            )
            _render_openaq_charts(df_cached)


def _fetch_openaq_live(api_key: str) -> pd.DataFrame:
    """
    Fetches the latest NO₂ readings from OpenAQ v3 for IT, FR, DE.
    Returns a DataFrame or None on failure.
    """
    BASE_URL = "https://api.openaq.org/v3"
    headers  = {"X-API-Key": api_key}
    COUNTRIES = ["IT", "FR", "DE"]
    all_rows = []

    for country in COUNTRIES:
        try:
            # Get locations for this country
            resp = requests.get(
                f"{BASE_URL}/locations",
                headers=headers,
                params={"iso": country, "limit": 20},
                timeout=10
            )
            if resp.status_code != 200:
                continue

            locations = resp.json().get("results", [])

            for loc in locations[:5]:   # Limit to 5 per country for speed
                loc_id = loc.get("id")
                sensors = loc.get("sensors", [])

                for sensor in sensors:
                    if sensor.get("parameter", {}).get("name", "").lower() != "no2":
                        continue
                    sensor_id = sensor.get("id")
                    if not sensor_id:
                        continue

                    # Fetch last 7 days of hourly data
                    date_to   = datetime.now(timezone.utc)
                    date_from = date_to - timedelta(days=7)

                    mresp = requests.get(
                        f"{BASE_URL}/sensors/{sensor_id}/hours",
                        headers=headers,
                        params={
                            "datetime_from": date_from.strftime("%Y-%m-%dT%H:%M:%SZ"),
                            "datetime_to":   date_to.strftime("%Y-%m-%dT%H:%M:%SZ"),
                            "limit": 100
                        },
                        timeout=10
                    )
                    if mresp.status_code != 200:
                        continue

                    for m in mresp.json().get("results", []):
                        period = m.get("period", {})
                        dt_info = period.get("datetimeTo", {})
                        all_rows.append({
                            "country":      country,
                            "location_id":  loc_id,
                            "city":         loc.get("locality", "Unknown"),
                            "datetime_utc": dt_info.get("utc"),
                            "no2_value":    m.get("value"),
                        })

                    time.sleep(0.2)   # Polite rate limiting

        except requests.exceptions.RequestException:
            continue  # Skip this country on network error

    if not all_rows:
        return None

    df = pd.DataFrame(all_rows)
    df["no2_value"]    = pd.to_numeric(df["no2_value"], errors="coerce")
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["no2_value", "datetime_utc"])
    df = df[(df["no2_value"] >= 0) & (df["no2_value"] <= 500)]
    return df


def _render_openaq_charts(df: pd.DataFrame):
    """Renders summary charts for OpenAQ NO₂ data."""
    if df is None or len(df) == 0:
        st.warning("No data to display.")
        return

    df["no2_value"] = pd.to_numeric(df["no2_value"], errors="coerce")
    df = df.dropna(subset=["no2_value"])

    # ── Summary metrics per country ───────────────────────────────
    if "country" in df.columns:
        st.markdown("### NO₂ summary by country")
        countries = df["country"].dropna().unique()
        metric_cols = st.columns(len(countries))

        for col, country in zip(metric_cols, sorted(countries)):
            subset = df[df["country"] == country]["no2_value"]
            with col:
                st.metric(
                    label=f"🇪🇺 {country} mean NO₂",
                    value=f"{subset.mean():.1f} µg/m³",
                    delta=f"max {subset.max():.0f} µg/m³",
                    delta_color="inverse"
                    # delta_color="inverse": red if positive (high max = bad)
                )

    # ── Bar chart: mean per city ───────────────────────────────────
    if "city" in df.columns:
        st.markdown("### Mean NO₂ by city")
        city_means = (
            df.groupby(["city", "country"])["no2_value"]
            .mean()
            .reset_index()
            .dropna()
            .sort_values("no2_value", ascending=False)
            .head(12)
        )

        if len(city_means) > 0:
            fig, ax = plt.subplots(figsize=(10, 5))
            country_palette = {"IT": "#E07B54", "FR": "#5B8DB8", "DE": "#6BAF92"}
            bar_colors = [country_palette.get(c, "#888") for c in city_means["country"]]

            ax.barh(
                city_means["city"],
                city_means["no2_value"],
                color=bar_colors,
                edgecolor="white",
                linewidth=0.4
            )
            ax.axvline(25, color="red", linestyle="--", linewidth=1.2,
                       label="WHO guideline (25 µg/m³)")
            ax.set_xlabel("Mean NO₂ (µg/m³)")
            ax.set_title("Mean NO₂ Concentration by City", fontweight="bold")
            ax.legend()

            legend_patches = [
                mpatches.Patch(color=v, label=k)
                for k, v in country_palette.items()
                if k in city_means["country"].values
            ]
            ax.legend(
                handles=legend_patches + [
                    plt.Line2D([0], [0], color="red", linestyle="--",
                               label="WHO guideline")
                ],
                fontsize=9
            )
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # ── Time series ───────────────────────────────────────────────
    if "datetime_utc" in df.columns and len(df) > 10:
        st.markdown("### NO₂ trend over time")
        fig2, ax2 = plt.subplots(figsize=(12, 4))

        for country, grp in df.groupby("country"):
            daily = (
                grp.set_index("datetime_utc")["no2_value"]
                .resample("6h").mean()
                # .resample("6h") groups by 6-hour windows.
                # This smooths the noisy hourly data for a cleaner chart.
            )
            ax2.plot(daily.index, daily.values, linewidth=1.5,
                     label=country, alpha=0.8)

        ax2.axhline(25, color="red", linestyle="--", linewidth=1,
                    alpha=0.6, label="WHO guideline")
        ax2.set_ylabel("NO₂ (µg/m³)")
        ax2.set_title("NO₂ Trend (6-hour averages)", fontweight="bold")
        ax2.legend()
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)


# ── Main router ───────────────────────────────────────────────────

def main():
    """
    Main entry point. Renders the sidebar and routes to the correct page.
    This function is called on every Streamlit rerun.
    """
    # ── Custom CSS ────────────────────────────────────────────────
    st.markdown("""
        <style>
            .block-container { padding-top: 1.5rem; }
            div[data-testid="stMetricValue"] { font-size: 1.4rem; }
        </style>
    """, unsafe_allow_html=True)

    # ── Startup diagnostics (helps debug Cloud file-not-found issues) ─
    # Check critical files exist before the app tries to load them.
    critical_files = [
        "models/xgb_classifier.pkl",
        "models/xgb_regressor.pkl",
        "data/processed/scaler.pkl",
        "data/processed/feature_cols.json",
    ]
    missing = [f for f in critical_files if not os.path.exists(f)]
    if missing:
        st.error(
            "**Missing files detected on startup:**\n"
            + "\n".join(f"- `{f}`" for f in missing)
            + "\n\nThese files must be committed to your GitHub repo. "
            "Check that `models/` and `data/processed/` are not in `.gitignore`."
        )
        st.stop()
        # st.stop() halts execution cleanly — shows the error without a full crash

    page = render_sidebar()

    # Route to the correct page based on sidebar selection.
    # Simple if/elif chain — Streamlit doesn't have a built-in router.
    if page == "🏠 Overview":
        page_overview()
    elif page == "🔮 Live Predictor":
        page_predictor()
    elif page == "🧠 SHAP Explainer":
        page_shap()
    elif page == "🌐 OpenAQ Live Data":
        page_openaq()


# ── Entry point ───────────────────────────────────────────────────
# Streamlit executes this file as a module on every rerun.
# main() must be called unconditionally — not guarded by __name__.
# We wrap in try/except so any startup crash is shown in the browser
# rather than silently killing the server (which produces "connection refused").
try:
    main()
except Exception as _startup_error:
    import traceback as _tb
    st.error("The app crashed on startup. Full traceback below:")
    st.code(_tb.format_exc())