import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import setup_path  # noqa: F401

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st

from app.utils import (
    inject_styles,
    load_model_bundle,
    plain_english_explanation,
    risk_category,
)
from src.config import COLORS, FEATURE_COLUMNS, FEATURE_LABELS
from src.data_pipeline import applicant_to_features

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

st.title("Risk Predictor")
st.markdown("Enter applicant financial details to estimate **probability of loan default**.")

try:
    model, _ = load_model_bundle()
except FileNotFoundError:
    st.error("Model not found. Run `python -m src.train` from the project root first.")
    st.stop()

tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

with tab1:
    with st.form("predict_form"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.slider("Age", min_value=18, max_value=80, value=35)
            monthly_income = st.number_input("Monthly Income (₹)", min_value=0, max_value=200_000, value=50_000, step=1000)
            debt_ratio = st.slider("Debt-to-Income Ratio (%)", min_value=0, max_value=100, value=35)
            open_lines = st.slider("Number of Open Credit Lines", min_value=0, max_value=20, value=5)
            late_30 = st.slider("Late Payments (30–59 days)", min_value=0, max_value=10, value=0)

        with col2:
            late_60 = st.slider("Late Payments (60–89 days)", min_value=0, max_value=10, value=0)
            late_90 = st.slider("Late Payments (90+ days)", min_value=0, max_value=10, value=0)
            dependents = st.selectbox("Number of Dependents", ["0", "1", "2", "3", "4", "5+"])
            revolving_balance = st.number_input(
                "Total Revolving Balance (₹)",
                min_value=0,
                max_value=1_000_000,
                value=10_000,
                step=1000,
            )
            utilisation = st.slider("Revolving Utilisation Rate (%)", min_value=0, max_value=100, value=30)

        submitted = st.form_submit_button("Predict Default Risk", type="primary", use_container_width=True)

    if submitted:
        inputs = {
            "age": age,
            "MonthlyIncome": float(monthly_income),
            "DebtRatio": debt_ratio / 100,
            "NumberOfOpenCreditLinesAndLoans": open_lines,
            "NumberOfTime30-59DaysPastDueNotWorse": late_30,
            "NumberOfTime60-89DaysPastDueNotWorse": late_60,
            "NumberOfTimes90DaysLate": late_90,
            "NumberOfDependents": dependents,
            "TotalRevolvingBalance": float(revolving_balance),
            "RevolvingUtilizationOfUnsecuredLines": utilisation / 100,
        }
        features = applicant_to_features(inputs)
        probability = float(model.predict_proba(features)[0, 1])
        category, _ = risk_category(probability)

        st.markdown("---")
        st.subheader("Prediction Results")

        res_col1, res_col2 = st.columns([1, 1])
        with res_col1:
            st.metric("Default Probability", f"{probability * 100:.1f}%")
            if category == "Low Risk":
                text_color = "#27500A"
                bg_color = "#EAF3DE"
            elif category == "Medium Risk":
                text_color = "#633806"
                bg_color = "#FAEEDA"
            else:
                text_color = "#A32D2D"
                bg_color = "#FCEBEB"
            st.markdown(
                f'<div style="background:{bg_color}; color:{text_color}; padding:8px 16px; border-radius:8px; font-weight:600; display:inline-block;">{category}</div>',
                unsafe_allow_html=True,
            )

        with res_col2:
            gauge_color = COLORS["success"]
            if probability >= 0.60:
                gauge_color = COLORS["danger"]
            elif probability >= 0.30:
                gauge_color = COLORS["warning"]

            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=probability * 100,
                    number={"suffix": "%"},
                    title={"text": "Default Risk Score"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": gauge_color},
                        "steps": [
                            {"range": [0, 30], "color": COLORS["success_bg"]},
                            {"range": [30, 60], "color": COLORS["warning_bg"]},
                            {"range": [60, 100], "color": COLORS["danger_bg"]},
                        ],
                        "threshold": {
                            "line": {"color": COLORS["text"], "width": 2},
                            "thickness": 0.8,
                            "value": probability * 100,
                        },
                    },
                )
            )
            fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk Interpretation Box
            if category == "Low Risk":
                st.success("✅ This applicant is likely to repay the loan.")
            elif category == "Medium Risk":
                st.warning("⚠️ This applicant has moderate risk. Review manually.")
            else:
                st.error("🚨 High default risk. Loan approval not recommended.")

        st.subheader("SHAP Explanation — Why this score?")
        with st.spinner("Computing SHAP values..."):
            explainer = shap.TreeExplainer(model)
            shap_output = explainer.shap_values(features)
            if isinstance(shap_output, list):
                shap_values = shap_output[1][0]
            else:
                shap_values = shap_output[0]

            explanation_df = {
                FEATURE_LABELS.get(name, name): value
                for name, value in zip(FEATURE_COLUMNS, shap_values)
            }
            sorted_items = sorted(explanation_df.items(), key=lambda x: abs(x[1]), reverse=True)

            labels = [item[0] for item in sorted_items[:10]]
            values = [item[1] for item in sorted_items[:10]]
            colors = [COLORS["danger"] if v > 0 else COLORS["success"] for v in values]

            shap_fig = go.Figure(
                go.Waterfall(
                    name="SHAP",
                    orientation="h",
                    y=labels[::-1],
                    x=values[::-1],
                    base=0,
                    increasing={"marker": {"color": COLORS["danger"]}},
                    decreasing={"marker": {"color": COLORS["success"]}},
                    totals={"marker": {"color": COLORS["primary"]}},
                )
            )
            shap_fig.update_layout(
                title="Feature contributions to default probability",
                xaxis_title="SHAP value (impact on log-odds)",
                height=420,
                plot_bgcolor=COLORS["background"],
                paper_bgcolor=COLORS["background"],
            )
            st.plotly_chart(shap_fig, use_container_width=True)
            st.markdown("🟢 **Green bars** push the score DOWN (reduce default risk) &nbsp;&nbsp; 🔴 **Red bars** push the score UP (increase default risk)")

        feature_inputs = features.iloc[0].to_dict()
        explanation = plain_english_explanation(shap_values, FEATURE_COLUMNS, feature_inputs)
        st.info(explanation)

        st.caption(
            "Red bars increase predicted default risk; green bars decrease it. "
            "SHAP values show each feature's contribution relative to the model baseline."
        )

with tab2:
    st.markdown("### 📋 Batch Prediction")
    st.markdown("Upload a CSV file containing applicant details to predict default risk in batch.")
    
    # Download sample CSV template
    sample_csv_data = (
        "age,MonthlyIncome,DebtRatio,NumberOfOpenCreditLinesAndLoans,"
        "NumberOfTime30-59DaysPastDueNotWorse,NumberOfTime60-89DaysPastDueNotWorse,"
        "NumberOfTimes90DaysLate,NumberOfDependents,TotalRevolvingBalance,"
        "RevolvingUtilizationOfUnsecuredLines\n"
        "35,5000.0,0.35,5,0,0,0,1,1500.0,0.30\n"
        "45,12000.0,0.45,8,1,0,0,2,6000.0,0.75\n"
        "28,2500.0,0.60,3,0,1,2,0,2000.0,0.95\n"
    )
    st.download_button(
        "📄 Download Sample CSV Template",
        data=sample_csv_data,
        file_name="sample_applicants.csv",
        mime="text/csv"
    )
    
    st.divider()
    
    uploaded_file = st.file_uploader("Upload CSV of applicants", type=["csv"])
    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            
            with st.spinner("Processing batch predictions..."):
                df_processed = df_upload.copy()
                
                # Compute TotalLatePayments if missing
                if "TotalLatePayments" not in df_processed.columns:
                    late_cols = [
                        "NumberOfTime30-59DaysPastDueNotWorse",
                        "NumberOfTime60-89DaysPastDueNotWorse",
                        "NumberOfTimes90DaysLate"
                    ]
                    for col in late_cols:
                        if col not in df_processed.columns:
                            df_processed[col] = 0
                    df_processed["TotalLatePayments"] = df_processed[late_cols].sum(axis=1)
                
                # Handle NumberOfDependents
                if "NumberOfDependents" in df_processed.columns:
                    df_processed["NumberOfDependents"] = pd.to_numeric(df_processed["NumberOfDependents"], errors='coerce').fillna(0)
                else:
                    df_processed["NumberOfDependents"] = 0.0
                
                # Check all model features
                for col in FEATURE_COLUMNS:
                    if col not in df_processed.columns:
                        df_processed[col] = 0.0
                
                # Select correct order of features
                X = df_processed[FEATURE_COLUMNS]
                
                # Predict
                probs = model.predict_proba(X)[:, 1]
                
                # Add columns
                df_result = df_upload.copy()
                df_result["Default_Probability"] = probs
                df_result["Default_Probability"] = df_result["Default_Probability"].round(4)
                
                def get_risk_cat(p):
                    if p < 0.30:
                        return "Low Risk"
                    if p < 0.60:
                        return "Medium Risk"
                    return "High Risk"
                
                df_result["Risk_Category"] = df_result["Default_Probability"].apply(get_risk_cat)
                
                st.success(f"Successfully processed {len(df_result)} records!")
                
                # Display Results
                st.dataframe(df_result, use_container_width=True)
                
                # Download Button
                result_csv = df_result.to_csv(index=False)
                st.download_button(
                    "⬇️ Download Results CSV",
                    data=result_csv,
                    file_name="risk_predictions.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Error processing CSV file: {e}")

st.divider()
st.markdown(
    """<div style='text-align:center; color:gray; font-size:13px;'>
    Built by Monika Dangi · BCA, SICSR Pune · 
    <a href='https://github.com/monikaffs'>GitHub</a> · 
    <a href='https://www.linkedin.com/in/monika-dangi/'>LinkedIn</a>
    </div>""", 
    unsafe_allow_html=True
)
