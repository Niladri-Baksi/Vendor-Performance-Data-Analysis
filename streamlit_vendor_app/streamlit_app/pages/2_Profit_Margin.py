"""
pages/2_Profit_Margin.py — Profit Margin regression (single XGBoost model)
with an interactive "try it yourself" predictor at the bottom.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils import (
    apply_css, render_sidebar, page_header, info_box,
    sec_label, load_data, get_regression_model,
    COLORS, PLOT_BASE,
)

st.set_page_config(page_title="Profit Margin Prediction", page_icon="📈", layout="wide")
apply_css()
render_sidebar()
page_header(
    "Profit Margin Prediction",
    "A single XGBoost regressor — predicts Profit Margin (%) from 9 vendor features",
    "📈",
)

info_box(
    "🎯 &nbsp;<strong>Goal:</strong> Given the purchase, pricing and volume details of a vendor-brand, "
    "predict what Profit Margin (%) it will generate. "
    "<strong>GrossProfit</strong> and <strong>SalestoPurchaseRatio</strong> are excluded from features — "
    "the first is a direct mathematical component of the target; the second is a near-perfect proxy for it."
)

# ─── Load model ────────────────────────────────────────────────────────────────
with st.spinner("Loading model…"):
    xgb, X_test, y_test, feat_names = get_regression_model()

preds = xgb.predict(X_test)
mae   = mean_absolute_error(y_test, preds)
rmse  = np.sqrt(mean_squared_error(y_test, preds))
r2    = r2_score(y_test, preds)

# ─── Metric strip ─────────────────────────────────────────────────────────────
sec_label("Model Performance on Test Set (20%)")
c1, c2, c3, c4 = st.columns(4)
c1.metric("R²",           f"{r2:.4f}")
c2.metric("MAE",          f"{mae:.2f} pp")
c3.metric("RMSE",         f"{rmse:.2f} pp")
c4.metric("Test Samples", f"{len(y_test):,}")

st.markdown("<hr style='border:none;border-top:1px solid rgba(249,115,22,0.15);margin:1.4rem 0'>", unsafe_allow_html=True)

# ─── Actual vs Predicted ──────────────────────────────────────────────────────
sec_label("Actual vs Predicted — Profit Margin")

lo = min(float(y_test.min()), float(preds.min()))
hi = max(float(y_test.max()), float(preds.max()))
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=y_test, y=preds,
    mode="markers",
    marker=dict(color=COLORS["primary"], size=4, opacity=0.45),
    hovertemplate="Actual: %{x:.1f}%<br>Predicted: %{y:.1f}%<extra></extra>",
))
fig.add_trace(go.Scatter(
    x=[lo, hi], y=[lo, hi],
    mode="lines",
    line=dict(color="rgba(255,255,255,0.3)", dash="dash", width=1.5),
    name="Perfect fit", hoverinfo="skip",
))
fig.update_layout(
    title=f"XGBoost  |  R² = {r2:.4f}",
    xaxis_title="Actual Profit Margin (%)",
    yaxis_title="Predicted Profit Margin (%)",
    showlegend=False,
    height=420,
    **PLOT_BASE,
)
st.plotly_chart(fig, use_container_width=True)

# ─── Feature importance ───────────────────────────────────────────────────────
st.markdown("<hr style='border:none;border-top:1px solid rgba(249,115,22,0.15);margin:1.4rem 0'>", unsafe_allow_html=True)
sec_label("Feature Importances")

feat_imp = (
    pd.Series(xgb.feature_importances_, index=feat_names)
    .sort_values(ascending=True)
)

fig_imp = go.Figure(go.Bar(
    x=feat_imp.values,
    y=feat_imp.index,
    orientation="h",
    marker=dict(
        color=feat_imp.values,
        colorscale=[[0,"rgba(249,115,22,0.3)"],[1,COLORS["primary"]]],
    ),
    hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
))
fig_imp.update_layout(
    xaxis_title="Importance Score",
    height=420,
    **PLOT_BASE,
)
st.plotly_chart(fig_imp, use_container_width=True)

# ─── Interactive predictor ────────────────────────────────────────────────────
st.markdown("<hr style='border:none;border-top:1px solid rgba(249,115,22,0.15);margin:1.4rem 0'>", unsafe_allow_html=True)
sec_label("Try It Yourself — Predict Profit Margin for a New Vendor")

info_box(
    "Adjust the sliders below. The model uses 9 features — 7 raw + 2 derived "
    "(<strong>PriceMarkup</strong> and <strong>OrderSize</strong>, computed automatically)."
)

df = load_data()

with st.form("reg_predict_form"):
    a, b, c, d = st.columns(4)
    purchase_price = a.slider("Purchase Price ($)",  0.0,  150.0, float(df["PurchasePrice"].median()),  step=0.5)
    actual_price   = b.slider("Actual / Retail Price ($)", 0.0, 200.0, float(df["ActualPrice"].median()), step=0.5)
    volume         = c.slider("Volume (ml / units)", 100,  3000,  int(df["Volume"].median()),             step=50)
    purchase_qty   = d.slider("Total Purchase Qty",  0, 5000, int(df["TotalPurchaseQuantity"].median()), step=50)

    e, f, g = st.columns(3)
    sales_qty         = e.slider("Total Sales Qty",         0, 5000,    int(df["TotalSalesQuantity"].median()),    step=50)
    purchase_dollars  = f.slider("Total Purchase Dollars ($)", 0.0, 50000.0, float(df["TotalPurchaseDollars"].median()), step=500.0)
    sales_dollars     = g.slider("Total Sales Dollars ($)",    0.0, 50000.0, float(df["TotalSalesDollars"].median()),    step=500.0)

    submitted = st.form_submit_button("🔮 Predict Profit Margin", use_container_width=True)

if submitted:
    # Derived features, computed the same way as in the notebook
    price_markup = (actual_price - purchase_price) / max(purchase_price, 0.01)
    q_bounds = df["TotalPurchaseQuantity"].quantile([0.25, 0.5, 0.75]).values
    if purchase_qty <= q_bounds[0]:
        order_size = 0.0
    elif purchase_qty <= q_bounds[1]:
        order_size = 1.0
    elif purchase_qty <= q_bounds[2]:
        order_size = 2.0
    else:
        order_size = 3.0

    row = {
        "PurchasePrice"        : purchase_price,
        "ActualPrice"          : actual_price,
        "Volume"               : volume,
        "TotalPurchaseQuantity": purchase_qty,
        "TotalSalesQuantity"   : sales_qty,
        "TotalPurchaseDollars" : purchase_dollars,
        "TotalSalesDollars"    : sales_dollars,
        "PriceMarkup"          : price_markup,
        "OrderSize"            : order_size,
    }
    row = {f: row[f] for f in feat_names}  # keep exact training column order

    pred = float(xgb.predict(pd.DataFrame([row]))[0])

    if pred >= 35:
        color, tier, badge_cls = COLORS["high"],    "High Margin",   "badge-high"
    elif pred >= 15:
        color, tier, badge_cls = COLORS["medium"],  "Medium Margin", "badge-medium"
    else:
        color, tier, badge_cls = COLORS["low"],     "Low Margin",    "badge-low"

    st.markdown(f"""
    <div class="pred-box" style="border-color:rgba({','.join(str(int(color.lstrip('#')[i:i+2],16)) for i in (0,2,4))},0.4);">
        <p style="color:rgba(255,255,255,0.55);margin:0;font-size:0.82rem;text-transform:uppercase;letter-spacing:0.08em;">
            Predicted Profit Margin
        </p>
        <p style="font-size:3rem;font-weight:800;color:{color};margin:0.3rem 0;line-height:1;">
            {pred:.1f}%
        </p>
        <span class="badge {badge_cls}">{tier}</span>
    </div>
    """, unsafe_allow_html=True)
