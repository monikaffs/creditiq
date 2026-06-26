import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    CLEANED_DATA,
    COLORS,
    FEATURE_LABELS,
    METRICS,
    RAW_DATA,
    SCALER,
    XGB_MODEL,
)


def inject_styles() -> None:
    st.markdown(
        f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;700&display=swap');
            
            .stApp, [data-testid="stSidebar"] {{
                font-family: 'Outfit', sans-serif !important;
            }}
            
            h1, h2, h3, [data-testid="stSidebar"] h3 {{
                font-family: 'Space Grotesk', sans-serif !important;
                font-weight: 700 !important;
                letter-spacing: -0.02em;
            }}
            
            .stApp {{
                background-color: {COLORS["background"]};
            }}
            /* Hide the hover-to-copy link icon next to headings */
            [data-testid="stHeaderActionElements"],
            div[data-testid="StyledLinkIconContainer"] {{
                display: none !important;
            }}
            [data-testid="stMetricValue"] {{
                color: {COLORS["text"]};
            }}
            .metric-card {{
                background: {COLORS["card"]};
                border: 1px solid {COLORS["border"]};
                border-radius: 12px;
                padding: 1.25rem 1.5rem;
                margin-bottom: 0.5rem;
            }}
            .metric-card h3 {{
                margin: 0;
                font-size: 0.95rem;
                color: {COLORS["text"]};
                opacity: 0.75;
            }}
            .metric-card p {{
                margin: 0.35rem 0 0 0;
                font-size: 1.75rem;
                font-weight: 700;
                color: {COLORS["primary"]};
            }}
            .info-box {{
                background: {COLORS["card"]};
                border: 1px solid {COLORS["border"]};
                border-left: 4px solid {COLORS["primary"]};
                border-radius: 10px;
                padding: 1rem 1.25rem;
                color: {COLORS["text"]};
            }}
            .risk-low {{
                background: {COLORS["success_bg"]};
                border: 1px solid #C5DDB8;
                border-radius: 10px;
                padding: 1rem;
            }}
            .risk-medium {{
                background: {COLORS["warning_bg"]};
                border: 1px solid #E8D4A8;
                border-radius: 10px;
                padding: 1rem;
            }}
            .risk-high {{
                background: {COLORS["danger_bg"]};
                border: 1px solid #E8B4B4;
                border-radius: 10px;
                padding: 1rem;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_cleaned_data() -> pd.DataFrame:
    if CLEANED_DATA.exists():
        return pd.read_csv(CLEANED_DATA)
    return pd.read_csv(RAW_DATA)


@st.cache_data
def load_metrics() -> dict:
    with open(METRICS, encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def load_model_bundle():
    model = joblib.load(XGB_MODEL)
    scaler = joblib.load(SCALER)
    return model, scaler


def metric_card(label: str, value: str) -> None:
    st.markdown(
        f'<div class="metric-card"><h3>{label}</h3><p>{value}</p></div>',
        unsafe_allow_html=True,
    )


def risk_category(probability: float) -> tuple[str, str]:
    if probability < 0.30:
        return "Low Risk", COLORS["success_bg"]
    if probability < 0.60:
        return "Medium Risk", COLORS["warning_bg"]
    return "High Risk", COLORS["danger_bg"]


def plain_english_explanation(shap_values, feature_names: list[str], inputs: dict) -> str:
    contributions = list(zip(feature_names, shap_values))
    contributions.sort(key=lambda item: abs(item[1]), reverse=True)

    top_feature, top_value = contributions[0]
    label = FEATURE_LABELS.get(top_feature, top_feature)

    direction = "increasing" if top_value > 0 else "decreasing"
    input_value = inputs.get(top_feature)

    if top_feature == "DebtRatio" and input_value is not None:
        display_value = f"{input_value * 100:.0f}%"
    elif top_feature == "RevolvingUtilizationOfUnsecuredLines" and input_value is not None:
        display_value = f"{input_value * 100:.0f}%"
    elif isinstance(input_value, float):
        display_value = f"{input_value:,.0f}" if input_value > 100 else f"{input_value:.1f}"
    else:
        display_value = str(input_value)

    return (
        f"Your {label.lower()} of {display_value} is the biggest factor "
        f"{direction} your default risk."
    )
