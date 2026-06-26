from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

RAW_DATA = DATA_DIR / "cs-training.csv"
CLEANED_DATA = DATA_DIR / "cleaned_data.csv"

XGB_MODEL = MODELS_DIR / "xgboost_model.pkl"
SCALER = MODELS_DIR / "scaler.pkl"
METRICS = MODELS_DIR / "metrics.json"
FEATURE_NAMES = MODELS_DIR / "feature_names.json"
FEATURE_LABELS = MODELS_DIR / "feature_labels.json"

TARGET = "SeriousDlqin2yrs"

FEATURE_COLUMNS = [
    "age",
    "MonthlyIncome",
    "DebtRatio",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
    "NumberOfDependents",
    "TotalRevolvingBalance",
    "RevolvingUtilizationOfUnsecuredLines",
    "TotalLatePayments",
]

FEATURE_LABELS = {
    "age": "Age",
    "MonthlyIncome": "Monthly Income",
    "DebtRatio": "Debt-to-Income Ratio",
    "NumberOfOpenCreditLinesAndLoans": "Open Credit Lines",
    "NumberOfTime30-59DaysPastDueNotWorse": "Late Payments (30-59 days)",
    "NumberOfTime60-89DaysPastDueNotWorse": "Late Payments (60-89 days)",
    "NumberOfTimes90DaysLate": "Late Payments (90+ days)",
    "NumberOfDependents": "Number of Dependents",
    "TotalRevolvingBalance": "Total Revolving Balance",
    "RevolvingUtilizationOfUnsecuredLines": "Revolving Utilisation Rate",
    "TotalLatePayments": "Total Late Payments",
}

COLORS = {
    "background": "#F9F9F6",
    "card": "#FFFFFF",
    "border": "#E5E5E0",
    "primary": "#378ADD",
    "text": "#2C2C2A",
    "success_bg": "#EAF3DE",
    "warning_bg": "#FAEEDA",
    "danger_bg": "#FCEBEB",
    "success": "#3D8B37",
    "warning": "#BA7517",
    "danger": "#D85A5A",
}
