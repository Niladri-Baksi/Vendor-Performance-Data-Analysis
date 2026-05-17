"""
pages/2_Profit_Margin.py — Profit Margin regression (RF + XGBoost)
with an interactive "try it yourself" predictor at the bottom.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils import (
    apply_css, render_sidebar, page_header, info_box, warn_box,
    sec_label, load_data, get_regression_models,
    COLORS, PLOT_BASE, REGRESSION_FEATURES,
)

st.set_page_config(page_title="Profit Margin Prediction", page_icon="📈", layout="wide")
apply_css()
render_sidebar()
page_header(
    "Profit Margin Prediction",
    "XGBoost & Random Forest regression — predicts Profit Margin (%) from vendor features",
    "📈",
)

info_box(
    "🎯 &nbsp;<strong>Goal:</strong> Given the purchase, pricing and logistics details of a vendor-brand, "
    "predict what Profit Margin (%) it will generate. "
    "<strong>GrossProfit</strong> and <strong>SalestoPurchaseRatio</strong> are excluded from features — "
    "the first is a direct mathematical component of the target; the second dominated feature importance "
    "as a near-perfect proxy and masked all other signals."
)

# ─── Train / load models ──────────────────────────────────────────────────────
# with st.spinner("Training models (cached after first run)…"):
with st.spinner("Loading models…"):
    rf, xgb, X_test, y_test, feat_names = get_regression_models()

rf_preds  = rf.predict(X_test)
xgb_preds = xgb.predict(X_test)

def metrics(y_true, y_pred):
    return dict(
        mae  = mean_absolute_error(y_true, y_pred),
        rmse = np.sqrt(mean_squared_error(y_true, y_pred)),
        r2   = r2_score(y_true, y_pred),
    )

rf_m  = metrics(y_test, rf_preds)
xgb_m = metrics(y_test, xgb_preds)

# ─── Metric strip ─────────────────────────────────────────────────────────────
sec_label("Model Performance on Test Set (20%)")
c1, c2, c3, c4 = st.columns(4)
c1.metric("XGBoost R²",  f"{xgb_m['r2']:.4f}",   delta=f"RF: {rf_m['r2']:.4f}")
c2.metric("XGBoost MAE", f"{xgb_m['mae']:.2f} pp", delta=f"RF: {rf_m['mae']:.2f}")
c3.metric("XGBoost RMSE",f"{xgb_m['rmse']:.2f} pp",delta=f"RF: {rf_m['rmse']:.2f}")
c4.metric("Test Samples", f"{len(y_test):,}")

st.markdown("<hr style='border:none;border-top:1px solid rgba(124,110,250,0.15);margin:1.4rem 0'>", unsafe_allow_html=True)

# ─── Actual vs Predicted ──────────────────────────────────────────────────────
sec_label("Actual vs Predicted — Profit Margin")
col_l, col_r = st.columns(2)

for col, name, preds, color in [
    (col_l, "Random Forest",  rf_preds,  COLORS["info"]),
    (col_r, "XGBoost",        xgb_preds, COLORS["primary"]),
]:
    lo = min(float(y_test.min()), float(preds.min()))
    hi = max(float(y_test.max()), float(preds.max()))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_test, y=preds,
        mode="markers",
        marker=dict(color=color, size=4, opacity=0.45),
        hovertemplate="Actual: %{x:.1f}%<br>Predicted: %{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi],
        mode="lines",
        line=dict(color="rgba(255,255,255,0.3)", dash="dash", width=1.5),
        name="Perfect fit", hoverinfo="skip",
    ))
    fig.update_layout(
        title=f"{name}  |  R² = {(r2_score(y_test, preds)):.4f}",
        xaxis_title="Actual Profit Margin (%)",
        yaxis_title="Predicted Profit Margin (%)",
        showlegend=False,
        height=400,
        **PLOT_BASE,
    )
    col.plotly_chart(fig, use_container_width=True)

# ─── Feature importance ───────────────────────────────────────────────────────
st.markdown("<hr style='border:none;border-top:1px solid rgba(124,110,250,0.15);margin:1.4rem 0'>", unsafe_allow_html=True)
sec_label("XGBoost Feature Importances")

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
        colorscale=[[0,"rgba(124,110,250,0.3)"],[1,COLORS["primary"]]],
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
st.markdown("<hr style='border:none;border-top:1px solid rgba(124,110,250,0.15);margin:1.4rem 0'>", unsafe_allow_html=True)
sec_label("Try It Yourself — Predict Profit Margin for a New Vendor")

info_box(
    "Adjust the sliders below. Remaining features are pre-filled with dataset medians. "
    "The XGBoost model predicts the expected Profit Margin."
)

df = load_data()

with st.form("reg_predict_form"):
    a, b, c = st.columns(3)
    purchase_price = a.slider("Purchase Price ($)",  0.0,  150.0, float(df["PurchasePrice"].median()),  step=0.5)
    actual_price   = b.slider("Actual / Retail Price ($)", 0.0, 200.0, float(df["ActualPrice"].median()), step=0.5)
    volume         = c.slider("Volume (ml / units)", 100,  3000,  int(df["Volume"].median()),             step=50)

    d, e, f = st.columns(3)
    purchase_qty   = d.slider("Total Purchase Qty",  0, 5000, int(df["TotalPurchaseQuantity"].median()), step=50)
    freight_cost   = e.slider("Total Freight Cost ($)", 0.0, 5000.0, float(df["TotalFreightCost"].median()), step=50.0)
    sales_qty      = f.slider("Total Sales Qty",     0, 5000, int(df["TotalSalesQuantity"].median()),   step=50)

    submitted = st.form_submit_button("🔮 Predict Profit Margin", use_container_width=True)

if submitted:
    row = {col: df[col].median() for col in REGRESSION_FEATURES}
    row["PurchasePrice"]         = purchase_price
    row["ActualPrice"]           = actual_price
    row["Volume"]                = volume
    row["TotalPurchaseQuantity"] = purchase_qty
    row["TotalFreightCost"]      = freight_cost
    row["TotalSalesQuantity"]    = sales_qty
    row["PriceMarkup"]           = (actual_price - purchase_price) / max(purchase_price, 0.01)
    row["FreightPerUnit"]        = freight_cost / max(purchase_qty, 1)
    # Use the same quantile boundaries as training (pd.qcut q=4) for OrderSize
    q_bounds = df["TotalPurchaseQuantity"].quantile([0.25, 0.5, 0.75]).values
    if purchase_qty <= q_bounds[0]:
        row["OrderSize"] = 0.0
    elif purchase_qty <= q_bounds[1]:
        row["OrderSize"] = 1.0
    elif purchase_qty <= q_bounds[2]:
        row["OrderSize"] = 2.0
    else:
        row["OrderSize"] = 3.0
    # row["PriceMarkup"]           = (actual_price - purchase_price) / max(purchase_price, 0.01)
    # row["FreightPerUnit"]        = freight_cost / max(purchase_qty, 1)
    # row["OrderSize"]             = float(min(3, purchase_qty // max(int(df["TotalPurchaseQuantity"].quantile(0.25)), 1)))

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
