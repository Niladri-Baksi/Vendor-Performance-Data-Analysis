"""
pages/3_Performance_Class.py — High / Medium / Low performance classifier.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from utils import (
    apply_css, render_sidebar, page_header, info_box, sec_label,
    load_data, get_performance_labels, get_classifier_models,
    COLORS, PLOT_BASE, CLASSIFICATION_FEATURES,
)

st.set_page_config(page_title="Performance Classification", page_icon="🏷️", layout="wide")
apply_css()
render_sidebar()
page_header(
    "Performance Classification",
    "Labels vendors as High / Medium / Low performers using a composite KPI score",
    "🏷️",
)

info_box(
    "🏗️ &nbsp;<strong>Label construction:</strong> Each vendor-brand is assigned a normalised score "
    "= min-max(ProfitMargin) + min-max(StockTurnover). Quantile-cut into thirds gives "
    "<strong style='color:#2ECC71;'>High</strong>, "
    "<strong style='color:#F1C40F;'>Medium</strong>, "
    "<strong style='color:#E74C3C;'>Low</strong>. "
    "ProfitMargin, StockTurnover, and GrossProfit are then excluded from the classifier "
    "features to prevent label leakage."
)

df = load_data()
df = get_performance_labels(df)

# ─── Train models ─────────────────────────────────────────────────────────────
with st.spinner("Training classifiers (cached after first run)…"):
    rf, xgb, X_test, y_test, le, feat_names = get_classifier_models()

rf_preds  = rf.predict(X_test)
xgb_preds = xgb.predict(X_test)
class_names = le.classes_

# ─── Class distribution + label validation ────────────────────────────────────
sec_label("Label Distribution & Validation")
col_l, col_r = st.columns(2)

with col_l:
    counts = df["PerformanceCategory"].value_counts().reindex(["High","Medium","Low"])
    fig_pie = go.Figure(go.Pie(
        labels=counts.index,
        values=counts.values,
        hole=0.52,
        marker=dict(colors=[COLORS["high"], COLORS["medium"], COLORS["low"]]),
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Share: %{percent}<extra></extra>",
    ))
    fig_pie.update_layout(
        title="Vendor-brand tier split",
        showlegend=False,
        height=360,
        **PLOT_BASE,
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col_r:
    fig_box = go.Figure()
    for tier, color in [("High", COLORS["high"]), ("Medium", COLORS["medium"]), ("Low", COLORS["low"])]:
        subset = df[df["PerformanceCategory"] == tier]
        fig_box.add_trace(go.Box(
            y=subset["ProfitMargin"],
            name=tier,
            marker_color=color,
            boxmean="sd",
            hovertemplate="<b>%{x}</b><br>%{y:.1f}%<extra></extra>",
        ))
    fig_box.update_layout(
        title="ProfitMargin distribution by tier",
        yaxis_title="Profit Margin (%)",
        showlegend=False,
        height=360,
        **PLOT_BASE,
    )
    st.plotly_chart(fig_box, use_container_width=True)

st.markdown("<hr style='border:none;border-top:1px solid rgba(124,110,250,0.15);margin:1.4rem 0'>", unsafe_allow_html=True)

# ─── Model metrics ────────────────────────────────────────────────────────────
sec_label("Classifier Performance on Test Set (20%)")

tab_rf, tab_xgb = st.tabs(["🌲 Random Forest", "⚡ XGBoost"])

for tab, name, preds in [(tab_rf, "Random Forest", rf_preds), (tab_xgb, "XGBoost", xgb_preds)]:
    with tab:
        report = classification_report(y_test, preds, target_names=class_names, output_dict=True)
        report_df = pd.DataFrame(report).T.iloc[:-3]   # per-class rows only

        m1, m2, m3 = st.columns(3)
        m1.metric("Accuracy",        f"{report['accuracy']*100:.1f}%")
        m2.metric("Macro F1",        f"{report['macro avg']['f1-score']:.3f}")
        m3.metric("Weighted F1",     f"{report['weighted avg']['f1-score']:.3f}")

        ccol_l, ccol_r = st.columns([1, 1.1])

        # Confusion matrix
        with ccol_l:
            cm = confusion_matrix(y_test, preds)
            fig_cm = go.Figure(go.Heatmap(
                z=cm,
                x=class_names.tolist(),
                y=class_names.tolist(),
                colorscale="Purples",
                text=cm,
                texttemplate="%{text}",
                textfont=dict(size=14, color="white"),
                hovertemplate="Predicted: %{x}<br>Actual: %{y}<br>Count: %{z}<extra></extra>",
                showscale=False,
            ))
            fig_cm.update_layout(
                title=f"{name} — Confusion Matrix",
                xaxis_title="Predicted",
                yaxis_title="Actual",
                height=340,
                **PLOT_BASE,
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        # Per-class F1 / precision / recall
        with ccol_r:
            tiers = ["High", "Low", "Medium"] if "Low" in report else class_names.tolist()
            tier_data = {t: report.get(t, {}) for t in tiers}

            fig_bar = go.Figure()
            metric_colors = [COLORS["primary"], COLORS["info"], COLORS["high"]]
            for metric, color in zip(["precision","recall","f1-score"], metric_colors):
                fig_bar.add_trace(go.Bar(
                    name=metric.capitalize(),
                    x=list(tier_data.keys()),
                    y=[tier_data[t].get(metric, 0) for t in tier_data],
                    marker_color=color,
                    hovertemplate=f"<b>%{{x}}</b><br>{metric}: %{{y:.3f}}<extra></extra>",
                ))
            fig_bar.update_layout(
                barmode="group",
                title="Per-class metrics",
                yaxis_title="Score",
                height=340,
                **PLOT_BASE,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

# ─── Feature importance ───────────────────────────────────────────────────────
st.markdown("<hr style='border:none;border-top:1px solid rgba(124,110,250,0.15);margin:1.4rem 0'>", unsafe_allow_html=True)
sec_label("XGBoost Feature Importances — Classification")

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
        colorscale=[[0,"rgba(52,152,219,0.3)"],[1, COLORS["info"]]],
    ),
    hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
))
fig_imp.update_layout(xaxis_title="Importance Score", height=430, **PLOT_BASE)
st.plotly_chart(fig_imp, use_container_width=True)
