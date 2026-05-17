"""
pages/4_Risk_Detection.py — Rule-based + Isolation Forest risky vendor detection.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from utils import (
    apply_css, render_sidebar, page_header,
    info_box, warn_box, danger_box, sec_label,
    load_data, get_risk_flags, COLORS, PLOT_BASE,
)

st.set_page_config(page_title="Risk Detection", page_icon="⚠️", layout="wide")
apply_css()
render_sidebar()
page_header(
    "Risk Detection",
    "Flags vendors with both low profit margin and low stock turnover",
    "⚠️",
)

info_box(
    "🔍 &nbsp;Two complementary approaches are used: "
    "<strong>Rule-based</strong> (both metrics below 25th percentile — transparent, easy to explain) "
    "and <strong>Isolation Forest</strong> (statistical anomaly detection — catches subtler patterns "
    "across 9 financial features). Vendors flagged by <em>both</em> methods are the highest-priority cases."
)

# ─── Load data + flags ────────────────────────────────────────────────────────
df_raw = load_data()
with st.spinner("Running anomaly detection (cached after first run)…"):
    df = get_risk_flags(df_raw)

pm_thresh = df_raw["ProfitMargin"].quantile(0.25)
st_thresh = df_raw["StockTurnover"].quantile(0.25)

# ─── Threshold summary ────────────────────────────────────────────────────────
sec_label("Detection Thresholds")
t1, t2, t3, t4 = st.columns(4)
t1.metric("Margin threshold (25th pct)",   f"{pm_thresh:.1f}%")
t2.metric("Turnover threshold (25th pct)", f"{st_thresh:.3f}")
t3.metric("Contamination (IsoForest)",     "7%",   delta="of dataset")
t4.metric("IsoForest features",            "9")

st.markdown("<hr style='border:none;border-top:1px solid rgba(124,110,250,0.15);margin:1.4rem 0'>", unsafe_allow_html=True)

# ─── Method comparison metrics ────────────────────────────────────────────────
sec_label("Method Comparison")

n_rule   = df["RiskFlag_Rule"].sum()
n_iso    = df["RiskFlag_IsoForest"].sum()
n_both   = ((df["RiskFlag_Rule"] == 1) & (df["RiskFlag_IsoForest"] == 1)).sum()
n_only_r = ((df["RiskFlag_Rule"] == 1) & (df["RiskFlag_IsoForest"] == 0)).sum()
n_only_i = ((df["RiskFlag_Rule"] == 0) & (df["RiskFlag_IsoForest"] == 1)).sum()

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Rule-based risky",          f"{n_rule:,}",  delta=f"{n_rule/len(df)*100:.1f}%")
m2.metric("IsoForest risky",           f"{n_iso:,}",   delta=f"{n_iso/len(df)*100:.1f}%")
m3.metric("Flagged by both",           f"{n_both:,}",  delta="Highest priority")
m4.metric("Only rule-based",           f"{n_only_r:,}", delta="Obvious risk")
m5.metric("Only IsoForest",            f"{n_only_i:,}", delta="Subtle anomalies")

st.markdown("<hr style='border:none;border-top:1px solid rgba(124,110,250,0.15);margin:1.4rem 0'>", unsafe_allow_html=True)

# ─── Scatter plots ────────────────────────────────────────────────────────────
sec_label("Risk Zone Visualisation")
col_l, col_r = st.columns(2)

def _scatter(col, flag_col, title, flag_label, safe_label):
    df_plot = df.copy()
    df_plot["_label"] = df_plot[flag_col].map({1: flag_label, 0: safe_label})
    fig = px.scatter(
        df_plot,
        x="StockTurnover", y="ProfitMargin",
        color="_label",
        color_discrete_map={flag_label: COLORS["low"], safe_label: COLORS["high"]},
        opacity=0.45,
        hover_data={"VendorName": True, "Description": True,
                    "ProfitMargin": ":.1f", "StockTurnover": ":.3f",
                    "_label": False},
        title=title,
    )
    fig.add_vline(x=st_thresh, line_dash="dash", line_color="rgba(255,255,255,0.35)",
                  annotation_text=f"Turnover {st_thresh:.2f}", annotation_font_size=10)
    fig.add_hline(y=pm_thresh, line_dash="dash", line_color="rgba(255,255,255,0.35)",
                  annotation_text=f"Margin {pm_thresh:.1f}%", annotation_font_size=10)

    xlim = df["StockTurnover"].quantile(0.98)
    ylim_lo = df["ProfitMargin"].quantile(0.01)
    fig.update_xaxes(range=[0, xlim])
    fig.update_yaxes(range=[ylim_lo, 100])
    fig.update_layout(
        legend=dict(title="", orientation="h", yanchor="bottom", y=1.02),
        xaxis_title="Stock Turnover",
        yaxis_title="Profit Margin (%)",
        height=440,
        **PLOT_BASE,
    )
    col.plotly_chart(fig, use_container_width=True)

_scatter(col_l, "RiskFlag_Rule",      "Rule-Based Risk Zone",     "⚠ Risky", "✓ Safe")
_scatter(col_r, "RiskFlag_IsoForest", "Isolation Forest Anomalies","⚠ Risky", "✓ Safe")

# ─── Anomaly score distribution ───────────────────────────────────────────────
sec_label("Anomaly Score Distribution (Isolation Forest)")

score_thresh = df.loc[df["RiskFlag_IsoForest"] == 1, "AnomalyScore"].min()

fig_dist = go.Figure()
fig_dist.add_trace(go.Histogram(
    x=df.loc[df["RiskFlag_IsoForest"] == 0, "AnomalyScore"],
    name="Normal", marker_color=COLORS["high"], opacity=0.7, nbinsx=60,
))
fig_dist.add_trace(go.Histogram(
    x=df.loc[df["RiskFlag_IsoForest"] == 1, "AnomalyScore"],
    name="Risky",  marker_color=COLORS["low"],  opacity=0.85, nbinsx=60,
))
fig_dist.add_vline(
    x=score_thresh, line_dash="dash", line_color="rgba(255,255,255,0.55)",
    annotation_text="Risk threshold", annotation_font_size=11,
)
fig_dist.update_layout(
    barmode="overlay",
    xaxis_title="Anomaly Score (higher = more anomalous)",
    yaxis_title="Count",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    height=340,
    **PLOT_BASE,
)
st.plotly_chart(fig_dist, use_container_width=True)

# ─── Risky vendors table ──────────────────────────────────────────────────────
st.markdown("<hr style='border:none;border-top:1px solid rgba(124,110,250,0.15);margin:1.4rem 0'>", unsafe_allow_html=True)
sec_label("Risky Vendor Records")

view_cols = [
    "VendorName", "Description",
    "ProfitMargin", "StockTurnover", "GrossProfit",
    "TotalSalesDollars", "AnomalyScore",
    "RiskFlag_Rule", "RiskFlag_IsoForest",
]

tab_both, tab_iso, tab_rule = st.tabs([
    f"🚨 Both Methods ({n_both:,})",
    f"🔬 IsoForest Only ({n_only_i:,})",
    f"📏 Rule-Based Only ({n_only_r:,})",
])

for tab, mask in [
    (tab_both, (df["RiskFlag_Rule"] == 1) & (df["RiskFlag_IsoForest"] == 1)),
    (tab_iso,  (df["RiskFlag_Rule"] == 0) & (df["RiskFlag_IsoForest"] == 1)),
    (tab_rule, (df["RiskFlag_Rule"] == 1) & (df["RiskFlag_IsoForest"] == 0)),
]:
    with tab:
        sub = df[mask][view_cols].sort_values("AnomalyScore", ascending=False).reset_index(drop=True)
        st.dataframe(
            sub.style
               .format({
                   "ProfitMargin": "{:.1f}%",
                   "StockTurnover": "{:.3f}",
                   "GrossProfit": "${:,.0f}",
                   "TotalSalesDollars": "${:,.0f}",
                   "AnomalyScore": "{:.4f}",
               })
               .background_gradient(subset=["AnomalyScore"], cmap="Reds"),
            use_container_width=True,
            height=440,
        )

warn_box(
    "⚡ &nbsp;Vendors appearing in the <strong>'Both Methods'</strong> tab are the most critical — "
    "the rule-based check confirms obvious low margin + low turnover, and the Isolation Forest "
    "confirms they are statistically anomalous across all 9 financial dimensions."
)
