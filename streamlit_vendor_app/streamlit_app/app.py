"""
app.py — Home page of the Vendor Performance Analytics platform.
"""

import streamlit as st
import plotly.express as px
from utils import (
    apply_css, render_sidebar, page_header,
    info_box, sec_label, load_data, COLORS, PLOT_BASE
)

st.set_page_config(
    page_title = "Vendor Analytics",
    page_icon  = "🏪",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)
apply_css()
render_sidebar()

# ─── Banner ───────────────────────────────────────────────────────────────────
df = load_data()

st.markdown(f"""
<div style="
    background:linear-gradient(135deg,rgba(124,110,250,0.18) 0%,rgba(167,139,250,0.05) 100%);
    border:1px solid rgba(124,110,250,0.3);
    border-radius:20px;
    padding:2.2rem 2.5rem;
    margin-bottom:1.8rem;
">
    <h1 style="
        font-size:2.3rem;font-weight:800;margin:0;
        background:linear-gradient(135deg,#7C6EFA 20%,#A78BFA 80%);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    ">🏪 Vendor Performance Analytics</h1>
    <p style="color:rgba(255,255,255,0.55);margin:0.5rem 0 0;font-size:1rem;max-width:680px;">
        An ML-powered intelligence platform for analysing, classifying, and monitoring
        vendor performance across <strong style="color:rgba(255,255,255,0.8);">{len(df):,}</strong> vendor-brand records.
    </p>
</div>
""", unsafe_allow_html=True)

# ─── KPI strip ────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)

total_vendors  = df["VendorNumber"].nunique()
avg_margin     = df["ProfitMargin"].median()
total_gp       = df["GrossProfit"].sum()
risky_pct      = (
    ((df["ProfitMargin"] < df["ProfitMargin"].quantile(0.25)) &
     (df["StockTurnover"] < df["StockTurnover"].quantile(0.25))).mean() * 100
)

c1.metric("Total Unique Vendors",   f"{total_vendors:,}")
c2.metric("Median Profit Margin",   f"{avg_margin:.1f}%")
c3.metric("Total Gross Profit",     f"${total_gp/1e6:.1f}M")
c4.metric("Rule-Based Risky",       f"{risky_pct:.1f}%",  delta="of all vendors", delta_color="inverse")

st.markdown("<hr style='border:none;border-top:1px solid rgba(124,110,250,0.15);margin:1.6rem 0'>", unsafe_allow_html=True)

# ─── Two-column layout ────────────────────────────────────────────────────────
left, right = st.columns([1.05, 1], gap="large")

with left:
    sec_label("Platform Models")

    MODELS = [
        ("📊", "Data Overview",           "Distributions, correlations and a top-vendor explorer.",              "Low",    "white"),
        ("📈", "Profit Margin Prediction", "XGBoost regression — predicts margin % from vendor features.",        "Medium", "#A78BFA"),
        ("🏷️", "Performance Classification","High / Medium / Low tier assignment via a composite KPI score.",     "Medium", "#A78BFA"),
        ("⚠️", "Risk Detection",           "Rule-based + Isolation Forest flagging of low-margin / low-turnover vendors.", "High", "#F87171"),
    ]

    for icon, name, desc, complexity, col in MODELS:
        st.markdown(f"""
        <div style="
            background:rgba(26,29,46,0.7);
            border:1px solid rgba(124,110,250,0.17);
            border-radius:12px;
            padding:1rem 1.25rem;
            margin-bottom:0.65rem;
            transition:border-color 0.2s;
        ">
            <p style="margin:0;font-size:1rem;font-weight:600;color:{col};">{icon} {name}</p>
            <p style="margin:0.25rem 0 0;font-size:0.84rem;color:rgba(255,255,255,0.5);">{desc}</p>
        </div>
        """, unsafe_allow_html=True)

with right:
    sec_label("Profit Margin Distribution")

    fig = px.histogram(
        df, x="ProfitMargin", nbins=80,
        color_discrete_sequence=[COLORS["primary"]],
    )
    fig.add_vline(
        x=df["ProfitMargin"].median(), line_dash="dash",
        line_color=COLORS["high"],
        annotation_text=f"Median {df['ProfitMargin'].median():.1f}%",
        annotation_font_color=COLORS["high"],
    )
    fig.update_layout(
        title="All vendor-brand records",
        xaxis_title="Profit Margin (%)",
        yaxis_title="Count",
        showlegend=False,
        **PLOT_BASE,
    )
    fig.update_layout(height=380, margin=dict(l=20, r=10, t=48, b=20))
    st.plotly_chart(fig, use_container_width=True)

# ─── Getting started callout ──────────────────────────────────────────────────
st.markdown("<hr style='border:none;border-top:1px solid rgba(124,110,250,0.15);margin:1.6rem 0'>", unsafe_allow_html=True)
info_box(
    "👈 &nbsp; Use the sidebar to navigate between pages. "
    "Models are trained on first visit and <strong>cached for the rest of the session</strong> — "
    "subsequent visits to any model page are instant."
)
