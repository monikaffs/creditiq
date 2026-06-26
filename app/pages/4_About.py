import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import setup_path  # noqa: F401

import streamlit as st

from app.utils import inject_styles

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

st.title("About This Project")

st.markdown("""
<div style='background:#EAF3DE; padding:1.5rem; border-radius:12px; 
             border-left: 4px solid #3B6D11;'>
  <h2 style='color:#27500A; margin:0;'>Monika Dangi</h2>
  <p style='color:#3B6D11; margin:4px 0;'>BCA · Symbiosis Institute of Computer Studies and Research (SICSR), Pune</p>
  <p style='color:#3B6D11; margin:0;'>
    Machine Learning
  </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

st.subheader("Business Problem")
st.markdown(
    """
    Banks lose billions annually to loan defaults. This project builds an **interpretable credit risk
    scoring tool** that estimates the probability a borrower will seriously delinqu within two years,
    and explains *why* using SHAP — supporting faster, fairer, and more transparent lending decisions.
    """
)

st.subheader("Dataset")
st.markdown(
    """
    **Give Me Some Credit** — [Kaggle Competition](https://www.kaggle.com/c/GiveMeSomeCredit)

    - 150,000 training records
    - 11 original features + engineered payment and balance features
    - Target: `SeriousDlqin2yrs` (1 = default within 2 years)
    """
)

st.subheader("Tech Stack")
st.markdown(
    """
    ![Python](https://img.shields.io/badge/Python-3.10+-378ADD?style=flat-square&logo=python&logoColor=white)
    ![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-378ADD?style=flat-square)
    ![SHAP](https://img.shields.io/badge/SHAP-Explainability-378ADD?style=flat-square)
    ![Streamlit](https://img.shields.io/badge/Streamlit-Cloud-378ADD?style=flat-square&logo=streamlit&logoColor=white)
    ![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-378ADD?style=flat-square)
    """
)

col1, col2, _ = st.columns([1.2, 1.2, 3])
col1.link_button("🐙 GitHub Repository", "https://github.com/monikaffs", use_container_width=True)
col2.link_button("💼 LinkedIn Profile", "https://www.linkedin.com/in/monika-dangi/", use_container_width=True)

st.divider()
st.markdown(
    """<div style='text-align:center; color:gray; font-size:13px;'>
    Built by Monika Dangi · BCA, SICSR Pune · 
    <a href='https://github.com/monikaffs'>GitHub</a> · 
    <a href='https://www.linkedin.com/in/monika-dangi/'>LinkedIn</a>
    </div>""", 
    unsafe_allow_html=True
)
