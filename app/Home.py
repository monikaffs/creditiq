import setup_path  # noqa: F401

import streamlit as st

from app.utils import inject_styles, load_metrics, metric_card

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

st.markdown(
    '<h1 style="font-family: \'Times New Roman\', Times, Georgia, serif; font-weight: 700; margin-bottom: 0.5rem;">💳 CreditIQ — Loan Default Predictor</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    "Predict loan default probability and explain model decisions with SHAP — "
    "built for transparent credit risk assessment."
)

try:
    metrics = load_metrics()
    total_records = f"{metrics['dataset']['rows']:,}"
    default_rate = f"{metrics['dataset']['default_rate_pct']}%"
    feature_count = str(metrics["dataset"]["features"])
except FileNotFoundError:
    total_records = "150,000"
    default_rate = "6.68%"
    feature_count = "11"

def custom_metric_card(label: str, value: str, border_color: str, text_color: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card" style="border-left: 4px solid {border_color};">
            <h3>{label}</h3>
            <p style="color: {text_color};">{value}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

col1, col2, col3 = st.columns(3)
with col1:
    custom_metric_card("Total Records", total_records, border_color="#3D8B37", text_color="#3D8B37")
with col2:
    custom_metric_card("Default Rate", default_rate, border_color="#A32D2D", text_color="#A32D2D")
with col3:
    custom_metric_card("Features Used", feature_count, border_color="#378ADD", text_color="#378ADD")

st.info("💡 This app predicts the probability a loan applicant will default and explains *why* using SHAP — making AI decisions transparent for banks and lenders.")

st.markdown("---")

st.subheader("What is loan default?")
st.markdown(
    """
    **Loan default** occurs when a borrower fails to repay their loan according to the agreed terms
    within a defined period (here, serious delinquency within two years).

    For banks and lenders, predicting default risk early helps:
    - Reduce financial losses from bad loans
    - Price interest rates more accurately
    - Approve creditworthy applicants faster
    - Meet regulatory requirements for fair lending
    """
)

st.subheader("Explore the app")
st.markdown(
    """
    Use the sidebar to navigate:

    1. **EDA Dashboard** — class imbalance, distributions, correlations, and feature importance
    2. **Risk Predictor** — enter applicant details and get a probability score with SHAP explanations
    3. **Model Performance** — compare Logistic Regression, Random Forest, and XGBoost
    4. **About** — project background, tech stack, and resume-ready summary
    """
)

with st.expander("How to use this app"):
    st.markdown("""
    1. Go to **Risk Predictor** to predict a single applicant
    2. Upload a CSV on **Risk Predictor** for batch predictions
    3. Check **EDA Dashboard** for data insights
    4. See **Model Performance** for accuracy metrics
    """)

st.divider()
st.markdown(
    """<div style='text-align:center; color:gray; font-size:13px;'>
    Built by Monika Dangi · BCA, SICSR Pune · 
    <a href='https://github.com/monikaffs'>GitHub</a> · 
    <a href='https://www.linkedin.com/in/monika-dangi/'>LinkedIn</a>
    </div>""", 
    unsafe_allow_html=True
)
