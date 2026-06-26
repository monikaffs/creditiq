import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import setup_path  # noqa: F401

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st

from app.utils import inject_styles, load_cleaned_data, load_metrics
from src.config import COLORS, FEATURE_LABELS, TARGET

st.set_page_config(
    page_title="CreditIQ",
    page_icon="💳",
    layout="wide"
)

# Sidebar Branding
st.sidebar.markdown("### 💳 CreditIQ")
st.sidebar.markdown("**Credit Risk Predictor**  \n*ML app with SHAP explainability*")
st.sidebar.divider()
inject_styles()

st.title("📊 EDA Dashboard")
st.markdown("Exploratory analysis of the **Give Me Some Credit** dataset (150K loan applicants).")

df = load_cleaned_data()

st.markdown("### 📌 Class Imbalance")
col1, col2 = st.columns([2, 1])
with col1:
    counts = df[TARGET].value_counts().sort_index()
    fig = px.bar(
        x=["No Default", "Default"],
        y=counts.values,
        color=["No Default", "Default"],
        color_discrete_map={"No Default": COLORS["primary"], "Default": COLORS["danger"]},
        labels={"x": "Outcome", "y": "Count"},
        title="Class Imbalance (Outcome Distribution)",
    )
    fig.update_layout(
        showlegend=True,
        legend_title="Outcome",
        xaxis_title="Outcome",
        yaxis_title="Count",
        plot_bgcolor=COLORS["background"],
        paper_bgcolor=COLORS["background"]
    )
    st.plotly_chart(fig, use_container_width=True)
with col2:
    default_pct = df[TARGET].mean() * 100
    st.markdown(
        f"""
        <div class="info-box">
            <strong>💡 EDA Insights: Class Imbalance</strong>
            <ul style="margin-bottom:0; margin-top:8px;">
                <li>Only <strong>{default_pct:.2f}%</strong> of applicants defaulted within 2 years.</li>
                <li>The dataset is heavily imbalanced (~93% non-default).</li>
                <li>Models must prioritize recall and AUC over raw accuracy.</li>
                <li>We handled imbalance with <code>scale_pos_weight</code> in XGBoost.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()
st.markdown("### 📊 Age Distribution by Default Status")
fig, ax = plt.subplots(figsize=(10, 4))
for label, color in [(0, COLORS["primary"]), (1, COLORS["danger"])]:
    subset = df.loc[df[TARGET] == label, "age"]
    ax.hist(subset, bins=30, alpha=0.6, label="Default" if label else "No Default", color=color)
ax.set_title("Age Distribution by Default Status")
ax.set_xlabel("Age")
ax.set_ylabel("Count")
ax.legend()
ax.set_facecolor(COLORS["background"])
fig.patch.set_facecolor(COLORS["background"])
st.pyplot(fig)
st.markdown(
    """
    <div class="info-box">
        <strong>💡 EDA Insights: Age Dynamics</strong>
        <ul style="margin-bottom:0; margin-top:8px;">
            <li>Younger applicants (roughly 25–40) appear more frequently in the dataset.</li>
            <li>Default rates are relatively higher among younger borrowers.</li>
            <li>Age alone is not deterministic — it interacts with income and debt burden.</li>
            <li>Outliers above the 99th percentile were capped during preprocessing.</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()
st.markdown("### 💰 Monthly Income — Outlier Handling")
col1, col2 = st.columns(2)
with col1:
    fig = px.box(
        df,
        y="MonthlyIncome",
        color_discrete_sequence=[COLORS["primary"]],
        labels={"MonthlyIncome": "Monthly Income"},
        title="Capped Monthly Income Distribution",
    )
    fig.update_traces(name="Monthly Income")
    fig.update_layout(
        xaxis_title="Applicants",
        yaxis_title="Monthly Income (₹)",
        showlegend=True,
        plot_bgcolor=COLORS["background"],
        paper_bgcolor=COLORS["background"]
    )
    st.plotly_chart(fig, use_container_width=True)
with col2:
    st.markdown(
        """
        <div class="info-box">
            <strong>💡 EDA Insights: Income Distribution</strong>
            <ul style="margin-bottom:0; margin-top:8px;">
                <li>Monthly income has a long right tail with extreme values.</li>
                <li>Missing income values were imputed with the median (~$5,400).</li>
                <li>Values above the 99th percentile were capped to reduce skew.</li>
                <li>Income alone shows weak separation between default classes.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()
st.markdown("### 📈 Debt-to-Income Ratio Distribution")
fig = px.histogram(
    df,
    x="DebtRatio",
    nbins=50,
    color=TARGET,
    color_discrete_map={0: COLORS["primary"], 1: COLORS["danger"]},
    labels={"DebtRatio": "Debt Ratio", TARGET: "Default Status"},
    title="Debt-to-Income Ratio Distribution by Default Status",
)
fig.update_layout(
    xaxis_title="Debt-to-Income Ratio",
    yaxis_title="Count",
    showlegend=True,
    legend_title="Default Status",
    plot_bgcolor=COLORS["background"],
    paper_bgcolor=COLORS["background"]
)
st.plotly_chart(fig, use_container_width=True)
st.markdown(
    """
    <div class="info-box">
        <strong>💡 EDA Insights: Debt-to-Income Burden</strong>
        <ul style="margin-bottom:0; margin-top:8px;">
            <li>Higher debt ratios cluster among defaulters.</li>
            <li>Many applicants have debt ratios below 0.5 (50%).</li>
            <li>Extreme debt ratios were capped at the 99th percentile.</li>
            <li>Debt burden is one of the strongest behavioural risk signals.</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()
st.markdown("### 🌡️ Feature Correlation Heatmap")
numeric_cols = [
    "age",
    "MonthlyIncome",
    "DebtRatio",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
    "NumberOfDependents",
    "RevolvingUtilizationOfUnsecuredLines",
    "TotalLatePayments",
    TARGET,
]
corr = df[numeric_cols].corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=False, cmap="Blues", ax=ax, cbar=True)
ax.set_title("Feature Correlation Heatmap")
ax.set_xlabel("Features")
ax.set_ylabel("Features")
ax.set_facecolor(COLORS["background"])
fig.patch.set_facecolor(COLORS["background"])
st.pyplot(fig)
st.markdown(
    f"""
    <div class="info-box">
        <strong>💡 EDA Insights: Feature Correlations</strong>
        <ul style="margin-top:8px;">
            <li>Late-payment features are positively correlated with each other.</li>
            <li><code>TotalLatePayments</code> captures combined delinquency history.</li>
            <li>Debt ratio and utilisation show moderate positive correlation.</li>
            <li>No feature pair exhibits problematic multicollinearity for tree models.</li>
        </ul>
        <div style="background: {COLORS["success_bg"]}; border-left: 4px solid {COLORS["success"]}; padding: 10px 14px; border-radius: 6px; margin-top: 10px; color: #2C2C2A; font-weight: 500;">
            🔍 <strong>Key Insight:</strong> Late Payments (90+ days) shows the strongest positive correlation with default. Applicants with 3+ late payments are 4x more likely to default.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()
st.markdown("### 🏆 Top 10 XGBoost Feature Importances")
try:
    metrics = load_metrics()
    importance_df = pd.DataFrame(metrics["feature_importance"]).head(10)
    importance_df["label"] = importance_df["feature"].map(FEATURE_LABELS)
    fig = px.bar(
        importance_df,
        x="importance",
        y="label",
        orientation="h",
        color_discrete_sequence=[COLORS["primary"]],
        labels={"importance": "Importance Score", "label": "Feature"},
        title="Top 10 XGBoost Feature Importances",
    )
    fig.update_traces(name="Feature Importance")
    fig.update_layout(
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        showlegend=True,
        yaxis={"categoryorder": "total ascending"},
        plot_bgcolor=COLORS["background"],
        paper_bgcolor=COLORS["background"]
    )
    st.plotly_chart(fig, use_container_width=True)
except FileNotFoundError:
    st.info("Train the model first (`python -m src.train`) to view feature importances.")

st.markdown(
    f"""
    <div class="info-box">
        <strong>💡 EDA Insights: Feature Importances</strong>
        <ul style="margin-top:8px;">
            <li>Past delinquency history dominates predictive power.</li>
            <li>Revolving utilisation and debt ratio rank among top drivers.</li>
            <li>Demographics (age, dependents) contribute but less than payment behaviour.</li>
            <li>Tree-based importance aligns with domain intuition for credit risk.</li>
        </ul>
        <div style="background: {COLORS["warning_bg"]}; border-left: 4px solid {COLORS["warning"]}; padding: 10px 14px; border-radius: 6px; margin-top: 10px; color: #2C2C2A; font-weight: 500;">
            💡 <strong>Key Prediction Driver:</strong> Total Late Payments is the most predictive feature, followed by Age and Revolving Utilisation Rate. Monthly Income reduces risk when higher.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()
with st.expander("📋 Dataset Summary Stats"):
    st.dataframe(df.describe())

st.divider()
st.markdown(
    """<div style='text-align:center; color:gray; font-size:13px;'>
    Built by Monika Dangi · BCA, SICSR Pune · 
    <a href='https://github.com/monikaffs'>GitHub</a> · 
    <a href='https://www.linkedin.com/in/monika-dangi/'>LinkedIn</a>
    </div>""", 
    unsafe_allow_html=True
)
