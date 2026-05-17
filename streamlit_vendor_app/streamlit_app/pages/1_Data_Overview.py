"""
pages/1_Data_Overview.py — Dataset stats, distributions, and correlations.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

from utils import (
    apply_css, render_sidebar, page_header, info_box, sec_label,
    load_data, COLORS, PLOT_BASE,
)

st.set_page_config(page_title="Data Overview", page_icon="📊", layout="wide")
apply_css()
render_sidebar()
page_header("Data Overview", "Dataset shape, key distributions and correlations", "📊")

df = load_data()

# ─── Shape metrics ────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Records",       f"{len(df):,}")
c2.metric("Unique Vendors",      f"{df['VendorNumber'].nunique():,}")
c3.metric("Unique Brands",       f"{df['Brand'].nunique():,}")
c4.metric("Features Available",  f"{len(df.columns)}")

st.markdown("<hr style='border:none;border-top:1px solid rgba(124,110,250,0.15);margin:1.4rem 0'>", unsafe_allow_html=True)

# ─── Key distributions (2×2) ──────────────────────────────────────────────────
sec_label("Key Column Distributions")

DIST_COLS = {
    "ProfitMargin"      : "Profit Margin (%)",
    "StockTurnover"     : "Stock Turnover",
    "GrossProfit"       : "Gross Profit ($)",
    "SalestoPurchaseRatio": "Sales-to-Purchase Ratio",
}
DIST_COLORS = [COLORS["primary"], COLORS["info"], COLORS["high"], COLORS["warning"]]

fig_dist = make_subplots(
    rows=2, cols=2,
    subplot_titles=list(DIST_COLS.values()),
    vertical_spacing=0.14,
    horizontal_spacing=0.08,
)

positions = [(1,1),(1,2),(2,1),(2,2)]
for (col, label), (r, c), color in zip(DIST_COLS.items(), positions, DIST_COLORS):
    clipped = df[col].clip(
        lower=df[col].quantile(0.01),
        upper=df[col].quantile(0.99),
    )
    fig_dist.add_trace(
        go.Histogram(
            x=clipped, nbinsx=60,
            marker_color=color, opacity=0.85,
            name=label, showlegend=False,
            hovertemplate=f"{label}: %{{x:.2f}}<br>Count: %{{y}}<extra></extra>",
        ),
        row=r, col=c,
    )
    fig_dist.add_vline(
        x=clipped.median(), line_dash="dash",
        line_color="rgba(255,255,255,0.45)", line_width=1.2,
        row=r, col=c,
    )

fig_dist.update_layout(height=520, **PLOT_BASE)
fig_dist.update_layout(title_text="", margin=dict(l=20, r=20, t=40, b=20))
st.plotly_chart(fig_dist, use_container_width=True)

# ─── Correlation heatmap ──────────────────────────────────────────────────────
st.markdown("<hr style='border:none;border-top:1px solid rgba(124,110,250,0.15);margin:1.4rem 0'>", unsafe_allow_html=True)
sec_label("Correlation Heatmap")

CORR_COLS = [
    "PurchasePrice","ActualPrice","Volume",
    "TotalSalesDollars","TotalPurchaseDollars",
    "GrossProfit","ProfitMargin","StockTurnover",
    "SalestoPurchaseRatio","TotalFreightCost",
]

corr = df[CORR_COLS].corr().round(2)

fig_corr = go.Figure(go.Heatmap(
    z=corr.values,
    x=corr.columns.tolist(),
    y=corr.index.tolist(),
    colorscale="RdBu",
    zmid=0,
    text=corr.values.round(2),
    texttemplate="%{text}",
    textfont=dict(size=9),
    hoverongaps=False,
    hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>Correlation: %{z:.2f}<extra></extra>",
))
fig_corr.update_layout(
    height=480,
    xaxis=dict(tickangle=-35, tickfont=dict(size=10)),
    yaxis=dict(tickfont=dict(size=10)),
    **PLOT_BASE,
)
fig_corr.update_layout(margin=dict(l=20, r=20, t=30, b=20))
st.plotly_chart(fig_corr, use_container_width=True)

info_box(
    "🔑 &nbsp; Strong positive correlations appear between <strong>TotalSalesDollars</strong>, "
    "<strong>TotalPurchaseDollars</strong>, and <strong>GrossProfit</strong> — as expected. "
    "Notice that <strong>SalestoPurchaseRatio</strong> is strongly correlated with "
    "<strong>ProfitMargin</strong>, which is why it is excluded from the regression model features."
)

# ─── Top & Bottom vendors table ───────────────────────────────────────────────
st.markdown("<hr style='border:none;border-top:1px solid rgba(124,110,250,0.15);margin:1.4rem 0'>", unsafe_allow_html=True)
sec_label("Top & Bottom Performers by Gross Profit")

disp_cols = ["VendorName","Description","ProfitMargin","GrossProfit",
             "TotalSalesDollars","StockTurnover"]

tab1, tab2 = st.tabs(["🏆 Top 20 by Gross Profit", "⚠️ Bottom 20 by Gross Profit"])

with tab1:
    top = df.nlargest(20, "GrossProfit")[disp_cols].reset_index(drop=True)
    st.dataframe(
        top.style
            .format({"ProfitMargin": "{:.1f}%", "GrossProfit": "${:,.0f}",
                     "TotalSalesDollars": "${:,.0f}", "StockTurnover": "{:.3f}"})
            .background_gradient(subset=["GrossProfit"], cmap="Greens"),
        use_container_width=True,
        height=460,
    )

with tab2:
    bot = df.nsmallest(20, "GrossProfit")[disp_cols].reset_index(drop=True)
    st.dataframe(
        bot.style
            .format({"ProfitMargin": "{:.1f}%", "GrossProfit": "${:,.0f}",
                     "TotalSalesDollars": "${:,.0f}", "StockTurnover": "{:.3f}"})
            .background_gradient(subset=["GrossProfit"], cmap="Reds_r"),
        use_container_width=True,
        height=460,
    )
