"""
utils.py — Shared utilities for the Vendor Performance Analytics platform.
Models are loaded from /models/ (pre-trained in ML_Models.ipynb) rather than
trained at runtime — this makes cold-start < 2 s and is safe to deploy.

One algorithm per task (XGBoost regressor, XGBoost classifier, Isolation
Forest) — no model-comparison code to load or explain.

─── RETRAINING ────────────────────────────────────────────────────────────────
Run the last cell of ML_Models.ipynb to regenerate /models/*.pkl whenever the
dataset or hyperparameters change.  Commit the /models/ folder with the app.
───────────────────────────────────────────────────────────────────────────────
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ─── Paths ────────────────────────────────────────────────────────────────────

# Change CSV_PATH to a relative path when deploying (put the CSV in the repo root).
CSV_PATH   = r"C:\Niladri\empty\coding\projects\Vendor Performance Data Analysis\dataset\vendor_sales_summary.csv"
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

RANDOM_STATE = 42

# ─── Colors & plot defaults (warm palette) ────────────────────────────────────

COLORS = {
    "primary" : "#F97316",  # warm orange — main accent
    "high"    : "#22C55E",  # green  — kept for "good" status
    "medium"  : "#F59E0B",  # amber  — kept for "caution" status
    "low"     : "#EF4444",  # red    — kept for "bad" status
    "info"    : "#FB923C",  # light orange
    "warning" : "#F59E0B",
    "neutral" : "#A8A29E",
}

PLOT_BASE = dict(
    template     = "plotly_dark",
    paper_bgcolor= "rgba(0,0,0,0)",
    plot_bgcolor = "rgba(28,24,22,0.4)",
    font         = dict(color="#FAFAFA", size=12),
    margin       = dict(l=20, r=20, t=48, b=20),
    hoverlabel   = dict(
        bgcolor      = "#1F1A17",
        bordercolor  = "rgba(249,115,22,0.5)",
        font_size    = 12,
    ),
)

# ─── Feature lists (must match the pkl files exactly) ────────────────────────
# One shared feature set is used for BOTH regression and classification.

FEATURES          = joblib.load(os.path.join(MODELS_DIR, "features.pkl"))
ANOMALY_FEATURES  = joblib.load(os.path.join(MODELS_DIR, "anomaly_features.pkl"))

# ─── CSS ──────────────────────────────────────────────────────────────────────

def apply_css() -> None:
    st.markdown("""
    <style>
    #MainMenu, footer { visibility: hidden; }
    .block-container { padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1200px; }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #211A16 0%, #14100E 100%);
        border-right: 1px solid rgba(249,115,22,0.18);
    }
    [data-testid="stSidebarNavLink"] { border-radius:8px !important; margin-bottom:2px !important; }
    [data-testid="stSidebarNavLink"]:hover { background:rgba(249,115,22,0.12) !important; }
    [data-testid="stSidebarNavLink"][aria-current="page"] {
        background:rgba(249,115,22,0.2) !important;
        border-left:3px solid #F97316 !important;
    }

    [data-testid="metric-container"] {
        background: rgba(249,115,22,0.07);
        border: 1px solid rgba(249,115,22,0.22);
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        transition: all 0.22s ease;
    }
    [data-testid="metric-container"]:hover {
        background: rgba(249,115,22,0.13);
        border-color: rgba(249,115,22,0.48);
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.78rem !important;
        color: rgba(255,255,255,0.52) !important;
        text-transform: uppercase;
        letter-spacing: 0.07em;
    }
    [data-testid="stMetricValue"] { font-size:1.65rem !important; font-weight:700 !important; }

    .callout {
        padding: 0.75rem 1.1rem;
        border-radius: 0 10px 10px 0;
        font-size: 0.87rem;
        line-height: 1.65;
        margin: 0.6rem 0 1rem;
    }
    .callout-info    { background:rgba(251,146,60,0.09); border-left:3px solid #FB923C; }
    .callout-warn    { background:rgba(245,158,11,0.09); border-left:3px solid #F59E0B; }
    .callout-success { background:rgba(34,197,94,0.09);  border-left:3px solid #22C55E; }
    .callout-danger  { background:rgba(239,68,68,0.09);  border-left:3px solid #EF4444; }

    .badge { display:inline-block; padding:3px 13px; border-radius:20px; font-size:0.8rem; font-weight:600; letter-spacing:0.04em; }
    .badge-high   { background:rgba(34,197,94,0.18);  color:#22C55E; border:1px solid rgba(34,197,94,0.4); }
    .badge-medium { background:rgba(245,158,11,0.18); color:#F59E0B; border:1px solid rgba(245,158,11,0.4); }
    .badge-low    { background:rgba(239,68,68,0.18);  color:#EF4444; border:1px solid rgba(239,68,68,0.4); }
    .badge-risk   { background:rgba(239,68,68,0.18);  color:#EF4444; border:1px solid rgba(239,68,68,0.4); }
    .badge-safe   { background:rgba(34,197,94,0.18);  color:#22C55E; border:1px solid rgba(34,197,94,0.4); }

    .sec-label {
        font-size:0.78rem; font-weight:600; text-transform:uppercase;
        letter-spacing:0.09em; color:rgba(249,115,22,0.9); margin:1.6rem 0 0.5rem;
    }
    .pred-box {
        border-radius:14px; padding:1.4rem; text-align:center;
        margin-top:0.8rem; border:1px solid rgba(249,115,22,0.3);
        background:rgba(249,115,22,0.07);
    }
    .verdict-card { border-radius:14px; padding:1.3rem 1.5rem; text-align:center; border:1px solid; }
    [data-testid="stSidebarNav"] { display: none !important; }
    [data-testid="stExpander"] { border:1px solid rgba(249,115,22,0.18) !important; border-radius:10px !important; }
    </style>
    """, unsafe_allow_html=True)


# ─── Sidebar ──────────────────────────────────────────────────────────────────

def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("""
        <div style="padding:0.4rem 0 1rem;border-bottom:1px solid rgba(249,115,22,0.2);margin-bottom:1rem;">
            <h2 style="color:#F97316;font-size:1.15rem;margin:0;font-weight:700;">🏪 Vendor Analytics</h2>
            <p style="color:rgba(255,255,255,0.4);font-size:0.77rem;margin:0.25rem 0 0;">Performance Intelligence Platform</p>
        </div>
        """, unsafe_allow_html=True)

        df = load_data()
        st.markdown(f"**📁 Records:** `{len(df):,}`")
        st.markdown(f"**🏷️ Unique vendors:** `{df['VendorNumber'].nunique():,}`")
        st.markdown("---")

        st.markdown('<p class="sec-label">Navigate</p>', unsafe_allow_html=True)
        st.page_link("app.py",                       label="🏠 Home")
        st.page_link("pages/1_Data_Overview.py",     label="📊 Data Overview")
        st.page_link("pages/2_Profit_Margin.py",     label="📈 Profit Margin Predictor")
        st.page_link("pages/3_Performance_Class.py", label="🏷️ Performance Classifier")
        st.page_link("pages/4_Risk_Detection.py",    label="⚠️ Risk Detector")

        st.markdown("---")
        st.caption("➕ Add pages to `pages/` and a link above to extend.")


# ─── Page header & callout helpers ───────────────────────────────────────────

def page_header(title: str, subtitle: str, icon: str = "") -> None:
    st.markdown(f"""
    <div style="
        background:linear-gradient(135deg,rgba(249,115,22,0.13) 0%,rgba(251,146,60,0.04) 100%);
        border:1px solid rgba(249,115,22,0.25);border-radius:16px;
        padding:1.6rem 2rem;margin-bottom:1.6rem;">
        <h1 style="font-size:1.9rem;font-weight:700;margin:0;
            background:linear-gradient(135deg,#F97316,#FB923C);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            {icon} {title}</h1>
        <p style="color:rgba(255,255,255,0.5);margin:0.4rem 0 0;font-size:0.95rem;">{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)

def info_box(t)    : st.markdown(f'<div class="callout callout-info">{t}</div>',    unsafe_allow_html=True)
def warn_box(t)    : st.markdown(f'<div class="callout callout-warn">{t}</div>',    unsafe_allow_html=True)
def success_box(t) : st.markdown(f'<div class="callout callout-success">{t}</div>', unsafe_allow_html=True)
def danger_box(t)  : st.markdown(f'<div class="callout callout-danger">{t}</div>',  unsafe_allow_html=True)
def sec_label(t)   : st.markdown(f'<p class="sec-label">{t}</p>',                   unsafe_allow_html=True)


# ─── Data ────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading dataset…")
def load_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH, sep=";")
    df = df[df["TotalSalesDollars"] > 0].copy()
    df = df[df["ProfitMargin"] > -200].copy()
    cap = df["StockTurnover"].quantile(0.99)
    df["StockTurnover"] = df["StockTurnover"].clip(upper=cap)
    df["OrderSize"]     = pd.qcut(df["TotalPurchaseQuantity"], q=4, labels=False).astype(float)
    df["PriceMarkup"]   = (df["ActualPrice"] - df["PurchasePrice"]) / df["PurchasePrice"].clip(lower=0.01)
    return df.reset_index(drop=True)


def get_performance_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Recreate Performance labels the same simple way the notebook does:
    percentile-rank ProfitMargin + percentile-rank StockTurnover, then split
    into thirds. No scaler object needed — nothing to load or fit."""
    df = df.copy()
    df["CompositeScore"] = (
        df["ProfitMargin"].rank(pct=True) + df["StockTurnover"].rank(pct=True)
    )
    df["PerformanceCategory"] = pd.qcut(
        df["CompositeScore"], q=3, labels=["Low", "Medium", "High"]
    ).astype(str)
    return df


# ─── Model loaders (cached — each pkl is read once per session) ──────────────

@st.cache_resource(show_spinner="Loading regression model…")
def get_regression_model():
    """Load the pre-trained XGBoost regressor and its saved test split."""
    xgb = joblib.load(os.path.join(MODELS_DIR, "xgb_reg.pkl"))
    X_test, y_test = joblib.load(os.path.join(MODELS_DIR, "reg_test_split.pkl"))
    return xgb, X_test, y_test, FEATURES


@st.cache_resource(show_spinner="Loading classifier model…")
def get_classifier_model():
    """Load the pre-trained XGBoost classifier, label encoder, and test split."""
    xgb = joblib.load(os.path.join(MODELS_DIR, "xgb_clf.pkl"))
    le  = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
    X_test, y_test = joblib.load(os.path.join(MODELS_DIR, "clf_test_split.pkl"))
    return xgb, X_test, y_test, le, FEATURES


@st.cache_resource(show_spinner="Loading anomaly detector…")
def get_anomaly_detector():
    """Load IsoForest, its scaler, and the rule-based thresholds."""
    iso        = joblib.load(os.path.join(MODELS_DIR, "iso_forest.pkl"))
    scaler     = joblib.load(os.path.join(MODELS_DIR, "iso_scaler.pkl"))
    thresholds = joblib.load(os.path.join(MODELS_DIR, "iso_thresholds.pkl"))
    return iso, scaler, thresholds


# ─── Risk flags (computed on the dataset, cached) ────────────────────────────

@st.cache_data(show_spinner="Computing risk flags…")
def get_risk_flags(_df: pd.DataFrame) -> pd.DataFrame:
    iso, scaler, thr = get_anomaly_detector()
    df = _df.copy()
    df["RiskFlag_Rule"] = (
        (df["ProfitMargin"] < thr["pm"]) & (df["StockTurnover"] < thr["st"])
    ).astype(int)
    X_scaled                 = scaler.transform(df[ANOMALY_FEATURES].fillna(0))
    df["RiskFlag_IsoForest"] = (iso.predict(X_scaled) == -1).astype(int)
    df["AnomalyScore"]       = -iso.decision_function(X_scaled)
    return df


# ─── Single-row prediction helpers ───────────────────────────────────────────

def predict_risk(feature_values: dict) -> dict:
    """Score a single new vendor for risk. Keys = ANOMALY_FEATURES."""
    iso, scaler, thr = get_anomaly_detector()
    row       = np.array([[feature_values.get(f, 0) for f in ANOMALY_FEATURES]])
    X_scaled  = scaler.transform(row)
    iso_flag  = bool(iso.predict(X_scaled)[0] == -1)
    iso_score = float(-iso.decision_function(X_scaled)[0])
    rule_flag = (
        feature_values["ProfitMargin"] < thr["pm"] and
        feature_values["StockTurnover"] < thr["st"]
    )
    return {
        "rule_flag": rule_flag,
        "iso_flag" : iso_flag,
        "iso_score": iso_score,
        "pm_thresh": thr["pm"],
        "st_thresh": thr["st"],
    }


def predict_margin(feature_values: dict) -> float:
    """Predict ProfitMargin for a single new vendor. Keys = FEATURES."""
    xgb, _, _, feat_names = get_regression_model()
    row = pd.DataFrame([{f: feature_values.get(f, 0) for f in feat_names}]).astype(float)
    return float(xgb.predict(row)[0])


def predict_performance(feature_values: dict) -> dict:
    """Predict tier for a single new vendor. Keys = FEATURES."""
    xgb, _, _, le, feat_names = get_classifier_model()
    row   = pd.DataFrame([{f: feature_values.get(f, 0) for f in feat_names}]).astype(float)
    proba = xgb.predict_proba(row)[0]
    pred  = le.inverse_transform([int(np.argmax(proba))])[0]
    return {"predicted": pred, "proba": proba, "class_names": le.classes_}


# ─── END OF UTILS ─────────────────────────────────────────────────────────────
