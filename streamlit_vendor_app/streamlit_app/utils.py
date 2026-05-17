"""
utils.py — Shared utilities for the Vendor Performance Analytics Streamlit app.

All data loading, preprocessing, model training and UI helpers live here.
Every model function is decorated with @st.cache_resource so models train
only once per session, regardless of how many times the page reruns.

─── ADDING A NEW MODEL ────────────────────────────────────────────────────────
1. Write a @st.cache_resource function below (see existing ones as templates).
2. Create pages/N_Your_Model_Name.py and import the function from utils.
3. Streamlit auto-discovers it and adds it to the sidebar — no other changes.
───────────────────────────────────────────────────────────────────────────────
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier, IsolationForest
)
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    classification_report, confusion_matrix,
)
from xgboost import XGBRegressor, XGBClassifier

# ─── Constants ────────────────────────────────────────────────────────────────

CSV_PATH     = r"C:\Niladri\empty\coding\projects\Vendor Performance Data Analysis\dataset\vendor_sales_summary.csv"
RANDOM_STATE = 42

COLORS = {
    "primary" : "#7C6EFA",
    "high"    : "#2ECC71",
    "medium"  : "#F1C40F",
    "low"     : "#E74C3C",
    "info"    : "#3498DB",
    "warning" : "#F39C12",
    "neutral" : "#95A5A6",
}

# Features used for predicting ProfitMargin.
# GrossProfit dropped  — direct mathematical component of ProfitMargin.
# SalestoPurchaseRatio dropped — acts as a near-perfect proxy (dominates
#   feature importance and masks every other signal).
REGRESSION_FEATURES = [
    "PurchasePrice", "Volume", "ActualPrice",
    "TotalPurchaseQuantity", "TotalPurchaseDollars",
    "TotalSalesQuantity", "TotalSalesDollars",
    "TotalSalesPrice", "TotalExciseTax", "TotalFreightCost",
    "PriceMarkup", "FreightPerUnit", "TaxPerUnit", "OrderSize",
]

# Features for High/Medium/Low classification.
# ProfitMargin + StockTurnover dropped — they BUILD the label (leakage).
# GrossProfit dropped — too strongly correlated with label.
CLASSIFICATION_FEATURES = [
    "PurchasePrice", "Volume", "ActualPrice",
    "TotalPurchaseQuantity", "TotalPurchaseDollars",
    "TotalSalesQuantity", "TotalSalesDollars",
    "TotalSalesPrice", "TotalExciseTax", "TotalFreightCost",
    "SalestoPurchaseRatio",
    "PriceMarkup", "FreightPerUnit", "TaxPerUnit", "OrderSize",
]

# Features used in Isolation Forest anomaly detection.
ANOMALY_FEATURES = [
    "ProfitMargin", "StockTurnover",
    "GrossProfit", "SalestoPurchaseRatio",
    "TotalSalesDollars", "TotalPurchaseDollars",
    "FreightPerUnit", "TaxPerUnit", "PriceMarkup",
]

# ─── CSS ──────────────────────────────────────────────────────────────────────

def apply_css() -> None:
    st.markdown("""
    <style>
    /* ── Base ── */
    #MainMenu, footer { visibility: hidden; }
    .block-container { padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1200px; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #161826 0%, #0F1117 100%);
        border-right: 1px solid rgba(124,110,250,0.18);
    }

    /* ── Metric cards ── */
    [data-testid="metric-container"] {
        background: rgba(124,110,250,0.07);
        border: 1px solid rgba(124,110,250,0.22);
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        transition: all 0.22s ease;
    }
    [data-testid="metric-container"]:hover {
        background: rgba(124,110,250,0.13);
        border-color: rgba(124,110,250,0.48);
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.78rem !important;
        color: rgba(255,255,255,0.52) !important;
        text-transform: uppercase;
        letter-spacing: 0.07em;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.65rem !important;
        font-weight: 700 !important;
    }

    /* ── Callout boxes ── */
    .callout {
        padding: 0.75rem 1.1rem;
        border-radius: 0 10px 10px 0;
        font-size: 0.87rem;
        line-height: 1.65;
        margin: 0.6rem 0 1rem;
    }
    .callout-info    { background:rgba(52,152,219,0.09);  border-left:3px solid #3498DB; }
    .callout-warn    { background:rgba(241,196,15,0.09);  border-left:3px solid #F1C40F; }
    .callout-success { background:rgba(46,204,113,0.09);  border-left:3px solid #2ECC71; }
    .callout-danger  { background:rgba(231,76,60,0.09);   border-left:3px solid #E74C3C; }

    /* ── Badges ── */
    .badge {
        display: inline-block;
        padding: 2px 11px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.04em;
    }
    .badge-high   { background:rgba(46,204,113,0.18);  color:#2ECC71; border:1px solid rgba(46,204,113,0.4); }
    .badge-medium { background:rgba(241,196,15,0.18);  color:#F1C40F; border:1px solid rgba(241,196,15,0.4); }
    .badge-low    { background:rgba(231,76,60,0.18);   color:#E74C3C; border:1px solid rgba(231,76,60,0.4); }
    .badge-risk   { background:rgba(231,76,60,0.18);   color:#E74C3C; border:1px solid rgba(231,76,60,0.4); }
    .badge-safe   { background:rgba(46,204,113,0.18);  color:#2ECC71; border:1px solid rgba(46,204,113,0.4); }

    /* ── Section labels ── */
    .sec-label {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        color: rgba(124,110,250,0.85);
        margin: 1.6rem 0 0.5rem;
    }

    /* ── Prediction result box ── */
    .pred-box {
        border-radius: 14px;
        padding: 1.4rem;
        text-align: center;
        margin-top: 1rem;
        border: 1px solid rgba(124,110,250,0.3);
        background: rgba(124,110,250,0.07);
    }

    /* ── Expander border ── */
    [data-testid="stExpander"] {
        border: 1px solid rgba(124,110,250,0.18) !important;
        border-radius: 10px !important;
    }
    </style>
    """, unsafe_allow_html=True)


# ─── Sidebar ──────────────────────────────────────────────────────────────────

def render_sidebar(active: str = "") -> None:
    """Renders a consistent branded sidebar across all pages."""
    with st.sidebar:
        st.markdown("""
        <div style="padding:0.4rem 0 1rem;border-bottom:1px solid rgba(124,110,250,0.2);margin-bottom:1rem;">
            <h2 style="color:#7C6EFA;font-size:1.15rem;margin:0;font-weight:700;">
                🏪 Vendor Analytics
            </h2>
            <p style="color:rgba(255,255,255,0.4);font-size:0.77rem;margin:0.25rem 0 0;">
                Performance Intelligence Platform
            </p>
        </div>
        """, unsafe_allow_html=True)

        df = load_data()
        st.markdown(f"**📁 Records:** `{len(df):,}`")
        st.markdown(f"**🏷️ Unique vendors:** `{df['VendorNumber'].nunique():,}`")
        st.markdown("---")

        st.markdown("""
        **🔬 Models**
        - 📊 Data Overview
        - 📈 Profit Margin
        - 🏷️ Performance Class
        - ⚠️ Risk Detection
        """)

        st.markdown("---")
        st.caption("➕ Add new pages to `pages/` to extend the platform.")
        st.caption("Models are cached after first load.")


# ─── Page header helpers ──────────────────────────────────────────────────────

def page_header(title: str, subtitle: str, icon: str = "") -> None:
    st.markdown(f"""
    <div style="
        background:linear-gradient(135deg,rgba(124,110,250,0.13) 0%,rgba(167,139,250,0.04) 100%);
        border:1px solid rgba(124,110,250,0.25);
        border-radius:16px;
        padding:1.6rem 2rem;
        margin-bottom:1.6rem;
    ">
        <h1 style="
            font-size:1.9rem;font-weight:700;margin:0;
            background:linear-gradient(135deg,#7C6EFA,#A78BFA);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;
        ">{icon} {title}</h1>
        <p style="color:rgba(255,255,255,0.5);margin:0.4rem 0 0;font-size:0.95rem;">{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def info_box(text: str) -> None:
    st.markdown(f'<div class="callout callout-info">{text}</div>', unsafe_allow_html=True)

def warn_box(text: str) -> None:
    st.markdown(f'<div class="callout callout-warn">{text}</div>', unsafe_allow_html=True)

def success_box(text: str) -> None:
    st.markdown(f'<div class="callout callout-success">{text}</div>', unsafe_allow_html=True)

def danger_box(text: str) -> None:
    st.markdown(f'<div class="callout callout-danger">{text}</div>', unsafe_allow_html=True)

def sec_label(text: str) -> None:
    st.markdown(f'<p class="sec-label">{text}</p>', unsafe_allow_html=True)


# ─── Plotly layout defaults ───────────────────────────────────────────────────

PLOT_BASE = dict(
    template     = "plotly_dark",
    paper_bgcolor= "rgba(0,0,0,0)",
    plot_bgcolor = "rgba(26,29,46,0.4)",
    font         = dict(color="#FAFAFA", size=12),
    margin       = dict(l=20, r=20, t=48, b=20),
    hoverlabel   = dict(bgcolor="#1A1D2E", bordercolor="rgba(124,110,250,0.5)", font_size=12),
)

def plotly_fig(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(height=height, **PLOT_BASE)
    return fig


# ─── Data loading & preprocessing ────────────────────────────────────────────

@st.cache_data(show_spinner="Loading dataset…")
def load_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH, sep=";")

    # Drop rows with no sales activity
    df = df[df["TotalSalesDollars"] > 0].copy()

    # Remove extreme ProfitMargin outliers (confirmed data artifacts in EDA)
    df = df[df["ProfitMargin"] > -200].copy()

    # Cap StockTurnover at 99th pct (tiny-volume artifacts)
    cap = df["StockTurnover"].quantile(0.99)
    df["StockTurnover"] = df["StockTurnover"].clip(upper=cap)

    # OrderSize — use labels=False to get integer codes, avoids category dtype
    # which XGBoost does not support
    df["OrderSize"] = pd.qcut(
        df["TotalPurchaseQuantity"], q=4, labels=False
    ).astype(float)

    # Derived features
    df["PriceMarkup"]    = (df["ActualPrice"] - df["PurchasePrice"]) / df["PurchasePrice"].clip(lower=0.01)
    df["FreightPerUnit"] = df["TotalFreightCost"] / df["TotalPurchaseQuantity"].clip(lower=1)
    df["TaxPerUnit"]     = df["TotalExciseTax"]   / df["TotalSalesQuantity"].clip(lower=1)

    return df.reset_index(drop=True)


def get_performance_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Adds PerformanceCategory column (High / Medium / Low) to df."""
    scaler = MinMaxScaler()
    scores = scaler.fit_transform(df[["ProfitMargin", "StockTurnover"]])
    df = df.copy()
    df["CompositeScore"] = scores[:, 0] + scores[:, 1]
    df["PerformanceCategory"] = pd.qcut(
        df["CompositeScore"], q=3, labels=["Low", "Medium", "High"]
    ).astype(str)
    return df


def get_risk_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Returns df with RiskFlag_Rule, RiskFlag_IsoForest, AnomalyScore columns."""
    df = df.copy()

    pm_thresh = df["ProfitMargin"].quantile(0.25)
    st_thresh = df["StockTurnover"].quantile(0.25)
    df["RiskFlag_Rule"] = (
        (df["ProfitMargin"] < pm_thresh) & (df["StockTurnover"] < st_thresh)
    ).astype(int)

    X_iso = df[ANOMALY_FEATURES].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_iso)

    iso = IsolationForest(n_estimators=200, contamination=0.07, random_state=RANDOM_STATE)
    iso.fit(X_scaled)
    df["RiskFlag_IsoForest"] = (iso.predict(X_scaled) == -1).astype(int)
    df["AnomalyScore"]       = -iso.decision_function(X_scaled)

    return df


# ─── Model training (cached as resources) ────────────────────────────────────

@st.cache_resource(show_spinner="Training regression models…")
def get_regression_models():
    """Returns RF model, XGB model, X_test, y_test, feature names."""
    df = load_data()
    X  = df[REGRESSION_FEATURES].astype(float)
    y  = df["ProfitMargin"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    rf = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)

    xgb = XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, random_state=RANDOM_STATE, verbosity=0,
    )
    xgb.fit(X_train, y_train)

    return rf, xgb, X_test, y_test, REGRESSION_FEATURES


@st.cache_resource(show_spinner="Training classification models…")
def get_classifier_models():
    """Returns RF, XGB, X_test, y_test, LabelEncoder."""
    df = load_data()
    df = get_performance_labels(df)

    le  = LabelEncoder()
    y   = le.fit_transform(df["PerformanceCategory"])
    X   = df[CLASSIFICATION_FEATURES].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)

    xgb = XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        eval_metric="mlogloss", random_state=RANDOM_STATE, verbosity=0,
    )
    xgb.fit(X_train, y_train)

    return rf, xgb, X_test, y_test, le, CLASSIFICATION_FEATURES


# ─── END OF UTILS — add new model functions above this line ──────────────────
