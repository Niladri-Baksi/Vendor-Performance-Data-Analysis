"""
pages/4_Risk_Detection.py — Risk flags (rule + IsoForest) + new-vendor risk predictor.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from utils import (
    apply_css, render_sidebar, page_header,
    info_box, warn_box, danger_box, sec_label,
    load_data, get_risk_flags, get_anomaly_detector,
    predict_risk, COLORS, PLOT_BASE, ANOMALY_FEATURES, success_box
)

st.set_page_config(page_title="Risk Detection", page_icon="⚠️", layout="wide")
apply_css()
render_sidebar()
page_header(
    "Risk Detection",
    "Flags vendors with low profit margin AND low stock turnover — and scores any new vendor instantly",
    "⚠️",
)

info_box(
    "🔍 &nbsp;Two complementary approaches: "
    "<strong>Rule-based</strong> (both metrics below 25th percentile — transparent) "
    "and <strong>Isolation Forest</strong> (statistical anomaly across 4 core financial features — catches subtler patterns). "
    "Vendors flagged by <em>both</em> are highest priority."
)

df_raw = load_data()
with st.spinner("Loading anomaly detector…"):
    df = get_risk_flags(df_raw)

_, _, thr = get_anomaly_detector()
pm_thresh = thr["pm"]
st_thresh = thr["st"]

# ─── Threshold strip ──────────────────────────────────────────────────────────
sec_label("Detection Thresholds")
t1, t2, t3, t4 = st.columns(4)
t1.metric("Margin threshold (25th pct)",   f"{pm_thresh:.1f}%")
t2.metric("Turnover threshold (25th pct)", f"{st_thresh:.3f}")
t3.metric("IsoForest contamination",       "10%")
t4.metric("IsoForest features",            f"{len(ANOMALY_FEATURES)}")

st.markdown("<hr style='border:none;border-top:1px solid rgba(249,115,22,0.15);margin:1.4rem 0'>", unsafe_allow_html=True)

# ─── Method comparison ────────────────────────────────────────────────────────
sec_label("Method Comparison")
n_rule   = int(df["RiskFlag_Rule"].sum())
n_iso    = int(df["RiskFlag_IsoForest"].sum())
n_both   = int(((df["RiskFlag_Rule"] == 1) & (df["RiskFlag_IsoForest"] == 1)).sum())
n_only_r = int(((df["RiskFlag_Rule"] == 1) & (df["RiskFlag_IsoForest"] == 0)).sum())
n_only_i = int(((df["RiskFlag_Rule"] == 0) & (df["RiskFlag_IsoForest"] == 1)).sum())

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Rule-based risky",  f"{n_rule:,}",   delta=f"{n_rule/len(df)*100:.1f}%")
m2.metric("IsoForest risky",   f"{n_iso:,}",    delta=f"{n_iso/len(df)*100:.1f}%")
m3.metric("Flagged by both",   f"{n_both:,}",   delta="Highest priority")
m4.metric("Only rule-based",   f"{n_only_r:,}", delta="Obvious risk")
m5.metric("Only IsoForest",    f"{n_only_i:,}", delta="Subtle anomalies")

st.markdown("<hr style='border:none;border-top:1px solid rgba(249,115,22,0.15);margin:1.4rem 0'>", unsafe_allow_html=True)

# ─── Scatter plots ────────────────────────────────────────────────────────────
sec_label("Risk Zone Visualisation")
col_l, col_r = st.columns(2)

def _scatter(col, flag_col, title):
    df_p = df.copy()
    df_p["_label"] = df_p[flag_col].map({1: "⚠ Risky", 0: "✓ Safe"})
    fig = px.scatter(
        df_p, x="StockTurnover", y="ProfitMargin", color="_label",
        color_discrete_map={"⚠ Risky": COLORS["low"], "✓ Safe": COLORS["high"]},
        opacity=0.45,
        hover_data={"VendorName": True, "Description": True,
                    "ProfitMargin": ":.1f", "StockTurnover": ":.3f", "_label": False},
        title=title,
    )
    fig.add_vline(x=st_thresh, line_dash="dash", line_color="rgba(255,255,255,0.35)",
                  annotation_text=f"Turnover {st_thresh:.2f}", annotation_font_size=10)
    fig.add_hline(y=pm_thresh, line_dash="dash", line_color="rgba(255,255,255,0.35)",
                  annotation_text=f"Margin {pm_thresh:.1f}%", annotation_font_size=10)
    xlim = df["StockTurnover"].quantile(0.98)
    fig.update_xaxes(range=[0, xlim])
    fig.update_yaxes(range=[df["ProfitMargin"].quantile(0.01), 100])
    fig.update_layout(
        legend=dict(title="", orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
        xaxis_title="Stock Turnover",
        yaxis_title="Profit Margin (%)",
        height=440,
        margin=dict(l=20, r=20, t=48, b=70),
        **{k: v for k, v in PLOT_BASE.items() if k != "margin"},
    )
    col.plotly_chart(fig, use_container_width=True)

_scatter(col_l, "RiskFlag_Rule",      "Rule-Based Risk Zone")
_scatter(col_r, "RiskFlag_IsoForest", "Isolation Forest Anomalies")

# ─── Anomaly score distribution ───────────────────────────────────────────────
sec_label("Anomaly Score Distribution")
score_thresh = float(df.loc[df["RiskFlag_IsoForest"] == 1, "AnomalyScore"].min())
fig_dist = go.Figure()
fig_dist.add_trace(go.Histogram(x=df.loc[df["RiskFlag_IsoForest"]==0,"AnomalyScore"],
    name="Normal", marker_color=COLORS["high"], opacity=0.7, nbinsx=60))
fig_dist.add_trace(go.Histogram(x=df.loc[df["RiskFlag_IsoForest"]==1,"AnomalyScore"],
    name="Risky",  marker_color=COLORS["low"],  opacity=0.85, nbinsx=60))
fig_dist.add_vline(x=score_thresh, line_dash="dash", line_color="rgba(255,255,255,0.55)",
                   annotation_text="Risk threshold", annotation_font_size=11)
fig_dist.update_layout(barmode="overlay", xaxis_title="Anomaly Score (higher = more anomalous)",
                       yaxis_title="Count", legend=dict(orientation="h", yanchor="bottom", y=1.02),
                       height=320, **PLOT_BASE)
st.plotly_chart(fig_dist, use_container_width=True)

# ─── Risky vendor tables ──────────────────────────────────────────────────────
st.markdown("<hr style='border:none;border-top:1px solid rgba(249,115,22,0.15);margin:1.4rem 0'>", unsafe_allow_html=True)
sec_label("Risky Vendor Records")
view_cols = ["VendorName","Description","ProfitMargin","StockTurnover",
             "GrossProfit","TotalSalesDollars","AnomalyScore","RiskFlag_Rule","RiskFlag_IsoForest"]
tab_both, tab_iso, tab_rule = st.tabs([
    f"🚨 Both Methods ({n_both:,})",
    f"🔬 IsoForest Only ({n_only_i:,})",
    f"📏 Rule-Based Only ({n_only_r:,})",
])
for tab, mask in [
    (tab_both, (df["RiskFlag_Rule"]==1) & (df["RiskFlag_IsoForest"]==1)),
    (tab_iso,  (df["RiskFlag_Rule"]==0) & (df["RiskFlag_IsoForest"]==1)),
    (tab_rule, (df["RiskFlag_Rule"]==1) & (df["RiskFlag_IsoForest"]==0)),
]:
    with tab:
        sub = df[mask][view_cols].sort_values("AnomalyScore", ascending=False).reset_index(drop=True)
        st.dataframe(
            sub.style
               .format({"ProfitMargin": "{:.1f}%", "StockTurnover": "{:.3f}",
                        "GrossProfit": "${:,.0f}", "TotalSalesDollars": "${:,.0f}",
                        "AnomalyScore": "{:.4f}"})
               .background_gradient(subset=["AnomalyScore"], cmap="Reds"),
            use_container_width=True, height=400,
        )

st.markdown("<hr style='border:none;border-top:1px solid rgba(249,115,22,0.15);margin:1.4rem 0'>", unsafe_allow_html=True)

# ─── NEW-VENDOR RISK PREDICTOR ────────────────────────────────────────────────
sec_label("🔮 Score a New Vendor for Risk")
info_box(
    "Enter the financial profile of a new or hypothetical vendor. "
    "The app instantly returns both a <strong>rule-based verdict</strong> "
    "(direct threshold comparison) and an <strong>Isolation Forest score</strong> "
    f"(statistical comparison against all {len(df_raw):,} existing vendors)."
)

df_raw2 = load_data()

with st.form("risk_predict_form"):
    st.markdown("**Core Risk Metrics**")
    a, b = st.columns(2)
    profit_margin  = a.number_input("Profit Margin (%)",
                                    -100.0, 100.0,
                                    float(df_raw2["ProfitMargin"].median()), step=1.0,
                                    help=f"Dataset 25th pct = {pm_thresh:.1f}%")
    stock_turnover = b.number_input("Stock Turnover",
                                    0.0, 20.0,
                                    float(df_raw2["StockTurnover"].median()), step=0.05,
                                    help=f"Dataset 25th pct = {st_thresh:.3f}")

    st.markdown("**Financial Totals**")
    c1, c2, c3 = st.columns(3)
    gross_profit      = c1.number_input("Gross Profit ($)", min_value=-5e4, max_value=2e6, value=float(df_raw2["GrossProfit"].median()), step=500.0)
    sales_dollars     = c2.number_input("Total Sales Dollars ($)",    min_value=0.0, value=float(df_raw2["TotalSalesDollars"].median()),    step=500.0)
    purchase_dollars  = c3.number_input("Total Purchase Dollars ($)", min_value=0.0, value=float(df_raw2["TotalPurchaseDollars"].median()), step=500.0)

    sales_purchase_ratio = sales_dollars / max(purchase_dollars, 0.01)

    submitted = st.form_submit_button("⚠️ Assess Risk", use_container_width=True)

if submitted:
    feat_vals = {
        "ProfitMargin"        : profit_margin,
        "StockTurnover"       : stock_turnover,
        "GrossProfit"         : gross_profit,
        "SalestoPurchaseRatio": sales_purchase_ratio,
    }

    res = predict_risk(feat_vals)

    st.markdown("<br>", unsafe_allow_html=True)
    vc_l, vc_r = st.columns(2)

    def _verdict_card(col, method_name, flagged, extra_line):
        color  = COLORS["low"] if flagged else COLORS["high"]
        icon   = "⚠️" if flagged else "✅"
        label  = "RISKY" if flagged else "SAFE"
        col.markdown(f"""
        <div class="verdict-card" style="border-color:{color}55;background:{color}11;">
            <p style="color:rgba(255,255,255,0.5);font-size:0.8rem;text-transform:uppercase;
                      letter-spacing:0.07em;margin:0;">{method_name}</p>
            <p style="font-size:2.8rem;margin:0.3rem 0;line-height:1;">{icon}</p>
            <p style="font-size:1.5rem;font-weight:700;color:{color};margin:0;">{label}</p>
            <p style="color:rgba(255,255,255,0.45);font-size:0.82rem;margin:0.4rem 0 0;">{extra_line}</p>
        </div>
        """, unsafe_allow_html=True)

    _verdict_card(
        vc_l, "Rule-Based", res["rule_flag"],
        f"Margin {profit_margin:.1f}% {'<' if profit_margin < pm_thresh else '≥'} threshold {pm_thresh:.1f}% &nbsp;|&nbsp; "
        f"Turnover {stock_turnover:.3f} {'<' if stock_turnover < st_thresh else '≥'} threshold {st_thresh:.3f}"
    )
    _verdict_card(
        vc_r, "Isolation Forest", res["iso_flag"],
        f"Anomaly Score: {res['iso_score']:.4f} &nbsp;({'above' if res['iso_flag'] else 'below'} risk threshold)"
    )

    st.markdown("<br>", unsafe_allow_html=True)
    if res["rule_flag"] and res["iso_flag"]:
        danger_box("🚨 &nbsp;<strong>Both methods flag this vendor as RISKY.</strong> Highest priority — "
                   "the rule-based threshold confirms low margin + low turnover, and Isolation Forest "
                   "confirms statistical anomaly across the core financial dimensions.")
    elif res["rule_flag"] or res["iso_flag"]:
        warn_box("⚠️ &nbsp;<strong>One method flags this vendor.</strong> Monitor closely. "
                 "Consider reviewing pricing strategy and inventory management.")
    else:
        success_box("✅ &nbsp;<strong>Vendor appears financially healthy.</strong> "
                    "Both metrics are above thresholds and the Isolation Forest finds no anomaly.")
