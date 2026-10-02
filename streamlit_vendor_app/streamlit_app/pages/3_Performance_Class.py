"""
pages/3_Performance_Class.py — Performance classifier (single XGBoost model)
+ new-vendor predictor.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from utils import (
    apply_css, render_sidebar, page_header, info_box, sec_label,
    load_data, get_performance_labels, get_classifier_model,
    predict_performance, COLORS, PLOT_BASE,
)

st.set_page_config(page_title="Performance Classification", page_icon="🏷️", layout="wide")
apply_css()
render_sidebar()
page_header(
    "Performance Classification",
    "Labels vendors High / Medium / Low via a percentile-rank score — and classifies new vendors instantly",
    "🏷️",
)

info_box(
    "🏗️ &nbsp;<strong>Label construction:</strong> each vendor is ranked as a percentile on "
    "ProfitMargin and on StockTurnover, the two ranks are summed, and the result is split into thirds. "
    "ProfitMargin, StockTurnover, and GrossProfit are <strong>excluded from classifier features</strong> "
    "to prevent label leakage — the same 9-feature set used for the regression model is reused here."
)

df = load_data()
df = get_performance_labels(df)

with st.spinner("Loading classifier…"):
    xgb, X_test, y_test, le, feat_names = get_classifier_model()

preds       = xgb.predict(X_test)
class_names = le.classes_   # alphabetical: ['High', 'Low', 'Medium']

# ─── Class distribution + box ─────────────────────────────────────────────────
sec_label("Label Distribution & Validation")
col_l, col_r = st.columns(2)

with col_l:
    counts = df["PerformanceCategory"].value_counts().reindex(["High","Medium","Low"])
    fig_pie = go.Figure(go.Pie(
        labels=counts.index, values=counts.values, hole=0.52,
        marker=dict(colors=[COLORS["high"], COLORS["medium"], COLORS["low"]]),
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Share: %{percent}<extra></extra>",
    ))
    fig_pie.update_layout(title="Vendor-brand tier split", showlegend=False, height=360, **PLOT_BASE)
    st.plotly_chart(fig_pie, use_container_width=True)

with col_r:
    fig_box = go.Figure()
    for tier, color in [("High", COLORS["high"]), ("Medium", COLORS["medium"]), ("Low", COLORS["low"])]:
        fig_box.add_trace(go.Box(
            y=df[df["PerformanceCategory"] == tier]["ProfitMargin"],
            name=tier, marker_color=color, boxmean="sd",
            hovertemplate="<b>%{x}</b><br>%{y:.1f}%<extra></extra>",
        ))
    fig_box.update_layout(title="ProfitMargin distribution by tier",
                          yaxis_title="Profit Margin (%)", showlegend=False, height=360, **PLOT_BASE)
    st.plotly_chart(fig_box, use_container_width=True)

st.markdown("<hr style='border:none;border-top:1px solid rgba(249,115,22,0.15);margin:1.4rem 0'>", unsafe_allow_html=True)

# ─── Model metrics ────────────────────────────────────────────────────────────
sec_label("Classifier Performance on Test Set (20%)")

report = classification_report(y_test, preds, target_names=class_names, output_dict=True)
m1, m2, m3 = st.columns(3)
m1.metric("Accuracy",    f"{report['accuracy']*100:.1f}%")
m2.metric("Macro F1",    f"{report['macro avg']['f1-score']:.3f}")
m3.metric("Weighted F1", f"{report['weighted avg']['f1-score']:.3f}")

cl, cr = st.columns([1, 1.1])

with cl:
    cm = confusion_matrix(y_test, preds)
    fig_cm = go.Figure(go.Heatmap(
        z=cm, x=class_names.tolist(), y=class_names.tolist(),
        colorscale="Oranges", text=cm, texttemplate="%{text}",
        textfont=dict(size=14, color="white"), showscale=False,
        hovertemplate="Predicted: %{x}<br>Actual: %{y}<br>Count: %{z}<extra></extra>",
    ))
    fig_cm.update_layout(title="XGBoost — Confusion Matrix",
                         xaxis_title="Predicted", yaxis_title="Actual", height=340, **PLOT_BASE)
    st.plotly_chart(fig_cm, use_container_width=True)

with cr:
    tiers         = ["High", "Medium", "Low"]
    metrics_list  = ["precision", "recall", "f1-score"]
    metric_labels = ["Precision", "Recall", "F1-Score"]
    metric_colors = [COLORS["primary"], COLORS["info"], COLORS["high"]]

    fig_bar = go.Figure()
    fig_bar.add_shape(
        type="line", x0=-0.5, x1=2.5, y0=1.0, y1=1.0,
        line=dict(color="rgba(255,255,255,0.2)", dash="dot", width=1.5),
    )
    for metric, label, color in zip(metrics_list, metric_labels, metric_colors):
        values = [report.get(t, {}).get(metric, 0) for t in tiers]
        fig_bar.add_trace(go.Bar(
            name=label, x=tiers, y=values,
            marker=dict(color=color, line=dict(color="rgba(255,255,255,0.15)", width=1)),
            text=[f"{v:.3f}" for v in values],
            textposition="outside",
            textfont=dict(size=11, color="rgba(255,255,255,0.85)"),
            hovertemplate=f"<b>%{{x}}</b><br>{label}: %{{y:.4f}}<extra></extra>",
            width=0.22,
        ))
    fig_bar.update_layout(
        barmode="group", bargroupgap=0.12,
        title=dict(text="Per-class Precision · Recall · F1", font=dict(size=13), x=0.02),
        yaxis=dict(title="Score", range=[0.93, 1.03], tickformat=".2f",
                   gridcolor="rgba(255,255,255,0.05)", zeroline=False),
        xaxis=dict(tickfont=dict(size=13, color="rgba(255,255,255,0.8)")),
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1,
                    font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
        height=340,
        **PLOT_BASE,
    )
    fig_bar.update_layout(margin=dict(l=10, r=10, t=58, b=20))
    st.plotly_chart(fig_bar, use_container_width=True)
    st.markdown(
        "<div style='font-size:0.8rem;color:rgba(255,255,255,0.38);"
        "text-align:center;margin-top:-0.4rem;'>"
        "Y-axis zoomed to 0.93–1.03 — scores are all in that tight band</div>",
        unsafe_allow_html=True,
    )

st.markdown("<hr style='border:none;border-top:1px solid rgba(249,115,22,0.15);margin:1.4rem 0'>", unsafe_allow_html=True)

# ─── Feature importance ───────────────────────────────────────────────────────
sec_label("Feature Importances")
feat_imp = pd.Series(xgb.feature_importances_, index=feat_names).sort_values(ascending=True)
fig_imp = go.Figure(go.Bar(
    x=feat_imp.values, y=feat_imp.index, orientation="h",
    marker=dict(color=feat_imp.values, colorscale=[[0,"rgba(251,146,60,0.3)"],[1,COLORS["primary"]]]),
    hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
))
fig_imp.update_layout(xaxis_title="Importance Score", height=430, **PLOT_BASE)
st.plotly_chart(fig_imp, use_container_width=True)

st.markdown("<hr style='border:none;border-top:1px solid rgba(249,115,22,0.15);margin:1.4rem 0'>", unsafe_allow_html=True)

# ─── NEW-VENDOR PREDICTOR ─────────────────────────────────────────────────────
sec_label("🔮 Classify a New Vendor")
info_box(
    "Enter the details of a new or hypothetical vendor-brand below. "
    "The XGBoost model will predict its performance tier and show the "
    "<strong>probability breakdown</strong> across all three classes. "
    "The same 9-feature set from the regression page is used here."
)

df_raw = load_data()

with st.form("clf_predict_form"):
    st.markdown("**Pricing & Product**")
    a, b, c = st.columns(3)
    purchase_price = a.number_input("Purchase Price ($)",   min_value=0.0, value=float(df_raw["PurchasePrice"].median()),  step=1.0)
    actual_price   = b.number_input("Actual / Retail ($)",  min_value=0.0, value=float(df_raw["ActualPrice"].median()),    step=1.0)
    volume         = c.number_input("Volume (ml / units)",  min_value=0,   value=int(df_raw["Volume"].median()),           step=50)

    st.markdown("**Volume**")
    d, e = st.columns(2)
    purchase_qty   = d.number_input("Total Purchase Qty",   min_value=0,   value=int(df_raw["TotalPurchaseQuantity"].median()),  step=100)
    sales_qty      = e.number_input("Total Sales Qty",      min_value=0,   value=int(df_raw["TotalSalesQuantity"].median()),    step=100)

    st.markdown("**Financial Totals**")
    f, g = st.columns(2)
    purchase_dollars = f.number_input("Total Purchase Dollars ($)", min_value=0.0, value=float(df_raw["TotalPurchaseDollars"].median()), step=500.0)
    sales_dollars    = g.number_input("Total Sales Dollars ($)",    min_value=0.0, value=float(df_raw["TotalSalesDollars"].median()),    step=500.0)

    submitted = st.form_submit_button("🏷️ Classify Vendor", use_container_width=True)

if submitted:
    price_markup = (actual_price - purchase_price) / max(purchase_price, 0.01)

    # OrderSize: which quartile does this purchase qty fall in?
    q1, q2, q3 = (df_raw["TotalPurchaseQuantity"].quantile(q) for q in [0.25, 0.5, 0.75])
    order_size = 0 if purchase_qty <= q1 else 1 if purchase_qty <= q2 else 2 if purchase_qty <= q3 else 3

    feat_vals = {
        "PurchasePrice"        : purchase_price,
        "ActualPrice"          : actual_price,
        "Volume"               : volume,
        "TotalPurchaseQuantity": purchase_qty,
        "TotalSalesQuantity"   : sales_qty,
        "TotalPurchaseDollars" : purchase_dollars,
        "TotalSalesDollars"    : sales_dollars,
        "PriceMarkup"          : price_markup,
        "OrderSize"            : float(order_size),
    }

    result = predict_performance(feat_vals)
    pred   = result["predicted"]
    proba  = result["proba"]
    cnames = result["class_names"]

    tier_cfg = {
        "High"  : (COLORS["high"],   "badge-high",   "✅ HIGH PERFORMER"),
        "Medium": (COLORS["medium"], "badge-medium", "🟡 MEDIUM PERFORMER"),
        "Low"   : (COLORS["low"],    "badge-low",    "⚠️ LOW PERFORMER"),
    }
    color, badge_cls, label = tier_cfg.get(pred, (COLORS["neutral"], "", pred))

    st.markdown(f"""
    <div class="pred-box" style="border-color:{color}55;">
        <p style="color:rgba(255,255,255,0.5);margin:0;font-size:0.8rem;text-transform:uppercase;letter-spacing:0.08em;">
            Predicted Performance Tier
        </p>
        <p style="font-size:2.4rem;font-weight:800;color:{color};margin:0.3rem 0;line-height:1.1;">{label}</p>
        <p style="color:rgba(255,255,255,0.4);font-size:0.82rem;margin:0;">
            XGBoost confidence: {max(proba)*100:.1f}%
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    sec_label("Probability Breakdown")

    display_order  = ["High", "Medium", "Low"]
    display_colors = [COLORS["high"], COLORS["medium"], COLORS["low"]]
    proba_ordered  = [proba[list(cnames).index(c)] for c in display_order]

    fig_proba = go.Figure(go.Bar(
        x=proba_ordered, y=display_order, orientation="h",
        marker=dict(color=display_colors),
        text=[f"{p*100:.1f}%" for p in proba_ordered],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Probability: %{x:.3f}<extra></extra>",
    ))
    fig_proba.update_layout(
        xaxis=dict(title="Probability", range=[0, 1.15], tickformat=".0%"),
        height=220, showlegend=False,
        **PLOT_BASE,
    )
    fig_proba.update_layout(margin=dict(l=20, r=80, t=20, b=20))
    st.plotly_chart(fig_proba, use_container_width=True)
