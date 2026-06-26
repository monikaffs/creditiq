import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import setup_path  # noqa: F401

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st

from sklearn.metrics import precision_score, recall_score, f1_score
from app.utils import inject_styles, load_metrics, load_cleaned_data, load_model_bundle
from src.config import COLORS

@st.cache_resource
def load_test_predictions():
    from sklearn.model_selection import train_test_split
    from src.data_pipeline import build_feature_matrix
    
    model, _ = load_model_bundle()
    df = load_cleaned_data()
    X, y = build_feature_matrix(df)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    y_prob = model.predict_proba(X_test)[:, 1]
    return y_test, y_prob

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

st.title("Model Performance")
st.markdown("Comparison of **Logistic Regression**, **Random Forest**, and **XGBoost** on a stratified 80/20 split.")

try:
    metrics = load_metrics()
except FileNotFoundError:
    st.error("Metrics not found. Run `python -m src.train` first.")
    st.stop()

comparison_df = pd.DataFrame(metrics["comparison"])
comparison_df = comparison_df.rename(
    columns={
        "model": "Model",
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
        "auc_roc": "AUC-ROC",
    }
)

st.subheader("Model Comparison")
display_df = comparison_df.copy()
display_df.loc[display_df["Model"] == "XGBoost", "Model"] = "XGBoost 🏆 Best"
for col in ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]:
    display_df[col] = display_df[col].map(lambda x: f"{x * 100:.2f}%")

xgb_metrics = metrics["xgboost"]

# Performance metric cards helper
def performance_metric_card(label: str, value: str, border_color: str, text_color: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card" style="border-left: 4px solid {border_color};">
            <h3>{label}</h3>
            <p style="color: {text_color};">{value}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

card_col1, card_col2, card_col3 = st.columns(3)
with card_col1:
    performance_metric_card("AUC-ROC (XGBoost)", f"{xgb_metrics['auc_roc'] * 100:.2f}%", "#378ADD", "#378ADD")
with card_col2:
    performance_metric_card("Recall (XGBoost)", f"{xgb_metrics['recall'] * 100:.2f}%", "#3D8B37", "#3D8B37")
with card_col3:
    performance_metric_card("F1 Score (XGBoost)", f"{xgb_metrics['f1'] * 100:.2f}%", "#BA7517", "#BA7517")

st.dataframe(display_df, use_container_width=True, hide_index=True)

st.warning("""
⚠️ **Why is Precision low (22-25%)?**
The dataset is heavily imbalanced — only 6.68% of applicants defaulted.
In such cases, raw Accuracy is misleading. We optimised for:
• **Recall** — catching as many true defaulters as possible
• **AUC-ROC** — overall model discrimination ability
This is standard practice in credit risk modelling.
""")
imbalance = metrics["class_imbalance"]

col1, col2 = st.columns(2)

with col1:
    st.subheader("Confusion Matrix (XGBoost)")
    cm = xgb_metrics["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="RdYlGn",
        xticklabels=["Predicted 0", "Predicted 1"],
        yticklabels=["Actual 0", "Actual 1"],
        ax=ax,
    )
    ax.set_facecolor(COLORS["background"])
    fig.patch.set_facecolor(COLORS["background"])
    st.pyplot(fig)

with col2:
    st.subheader("ROC Curve (XGBoost)")
    roc = xgb_metrics["roc_curve"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=roc["fpr"],
            y=roc["tpr"],
            mode="lines",
            name=f"AUC = {xgb_metrics['auc_roc']:.3f}",
            line={"color": COLORS["primary"], "width": 3},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random",
            line={"dash": "dash", "color": "#999"},
        )
    )
    fig.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        plot_bgcolor=COLORS["background"],
        paper_bgcolor=COLORS["background"],
    )
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Precision-Recall Curve (XGBoost)")
pr = xgb_metrics["pr_curve"]
pr_fig = px.line(
    x=pr["recall"],
    y=pr["precision"],
    labels={"x": "Recall", "y": "Precision"},
    title="Precision-Recall Curve (XGBoost)"
)
pr_fig.update_traces(line={"color": COLORS["primary"], "width": 3}, name="Precision-Recall")
pr_fig.update_layout(
    xaxis_title="Recall",
    yaxis_title="Precision",
    showlegend=True,
    plot_bgcolor=COLORS["background"],
    paper_bgcolor=COLORS["background"]
)
st.plotly_chart(pr_fig, use_container_width=True)

# Decision Threshold Tuner
st.subheader("🎛️ Classification Threshold Tuner")
st.markdown("Move the threshold to see how Precision and Recall trade off:")

threshold = st.slider("Decision threshold", 0.1, 0.9, 0.5, 0.01)

# Recalculate predictions at chosen threshold
y_test, y_prob = load_test_predictions()
y_pred_thresh = (y_prob >= threshold).astype(int)

# Show updated metrics in 3 columns
col_t1, col_t2, col_t3 = st.columns(3)
col_t1.metric("Precision", f"{precision_score(y_test, y_pred_thresh):.2%}")
col_t2.metric("Recall",    f"{recall_score(y_test, y_pred_thresh):.2%}")
col_t3.metric("F1 Score",  f"{f1_score(y_test, y_pred_thresh):.2%}")

st.caption("💡 Lower threshold = higher recall (catch more defaulters) "
           "but lower precision (more false alarms). Banks typically "
           "prefer higher recall to minimise financial loss.")

st.markdown("---")
st.subheader("Why XGBoost?")
st.markdown(
    f"""
    **XGBoost** was selected as the production model because it achieved the strongest balance of
    **AUC-ROC ({xgb_metrics['auc_roc'] * 100:.1f}%)** and **recall for defaulters ({xgb_metrics['recall'] * 100:.1f}%)**
    while remaining fast at inference time.

    - **Logistic Regression** provides a useful linear baseline but underfits complex interactions.
    - **Random Forest** performs well but is slower and slightly weaker on minority-class recall.
    - **XGBoost** handles missing values natively, supports class weighting, and works well with tabular credit data.
    """
)

st.subheader("Class Imbalance Handling")
st.markdown(
    f"""
    The dataset has a default rate of **{metrics['dataset']['default_rate_pct']}%**.
    Accuracy alone would be misleading because a naive model predicting "no default" every time
    would score ~93% accuracy.

    **Approach used:** `scale_pos_weight = {imbalance['scale_pos_weight']:.2f}` in XGBoost
    (ratio of non-default to default samples). Baseline models used `class_weight='balanced'`.

    {imbalance['method']}
    """
)

st.markdown(
    """
    **Interview tip:** For imbalanced credit datasets, prioritize **AUC-ROC**, **PR curve**, and
    **recall on defaulters** over headline accuracy.
    """
)

st.divider()
st.markdown(
    """<div style='text-align:center; color:gray; font-size:13px;'>
    Built by Monika Dangi · BCA, SICSR Pune · 
    <a href='https://github.com/monikaffs'>GitHub</a> · 
    <a href='https://www.linkedin.com/in/monika-dangi/'>LinkedIn</a>
    </div>""", 
    unsafe_allow_html=True
)
