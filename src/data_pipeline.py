import numpy as np
import pandas as pd

from src.config import CLEANED_DATA, FEATURE_COLUMNS, RAW_DATA, TARGET


OUTLIER_COLUMNS = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberRealEstateLoansOrLines",
    "TotalRevolvingBalance",
]


def load_raw_data(path=RAW_DATA) -> pd.DataFrame:
    df = pd.read_csv(path, on_bad_lines="skip")
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    for id_col in ("Id", "ID"):
        if id_col in df.columns:
            df = df.drop(columns=[id_col])
    return df


def cap_outliers(df: pd.DataFrame, columns: list[str], percentile: float = 99) -> pd.DataFrame:
    capped = df.copy()
    for col in columns:
        if col not in capped.columns:
            continue
        upper = capped[col].quantile(percentile / 100)
        capped[col] = capped[col].clip(upper=upper)
    return capped


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    median_income = data["MonthlyIncome"].median()

    data["MonthlyIncome"] = data["MonthlyIncome"].fillna(median_income)
    data["NumberOfDependents"] = data["NumberOfDependents"].fillna(0)

    data["TotalLatePayments"] = (
        data["NumberOfTime30-59DaysPastDueNotWorse"]
        + data["NumberOfTime60-89DaysPastDueNotWorse"]
        + data["NumberOfTimes90DaysLate"]
    )

    data["TotalRevolvingBalance"] = (
        data["RevolvingUtilizationOfUnsecuredLines"] * data["MonthlyIncome"]
    ).clip(lower=0)

    return data


def preprocess_data(save_cleaned: bool = True) -> pd.DataFrame:
    df = load_raw_data()
    df = engineer_features(df)
    df = cap_outliers(df, OUTLIER_COLUMNS)

    if save_cleaned:
        CLEANED_DATA.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(CLEANED_DATA, index=False)

    return df


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET].astype(int)
    return X, y


def applicant_to_features(inputs: dict) -> pd.DataFrame:
    dependents = inputs["NumberOfDependents"]
    if isinstance(dependents, str) and dependents.endswith("+"):
        dependents = int(dependents.replace("+", ""))

    debt_ratio = inputs["DebtRatio"]
    if debt_ratio > 1:
        debt_ratio = debt_ratio / 100

    utilisation = inputs["RevolvingUtilizationOfUnsecuredLines"]
    if utilisation > 1:
        utilisation = utilisation / 100

    late_30 = inputs["NumberOfTime30-59DaysPastDueNotWorse"]
    late_60 = inputs["NumberOfTime60-89DaysPastDueNotWorse"]
    late_90 = inputs["NumberOfTimes90DaysLate"]

    row = {
        "age": inputs["age"],
        "MonthlyIncome": inputs["MonthlyIncome"],
        "DebtRatio": debt_ratio,
        "NumberOfOpenCreditLinesAndLoans": inputs["NumberOfOpenCreditLinesAndLoans"],
        "NumberOfTime30-59DaysPastDueNotWorse": late_30,
        "NumberOfTime60-89DaysPastDueNotWorse": late_60,
        "NumberOfTimes90DaysLate": late_90,
        "NumberOfDependents": float(dependents),
        "TotalRevolvingBalance": inputs["TotalRevolvingBalance"],
        "RevolvingUtilizationOfUnsecuredLines": utilisation,
        "TotalLatePayments": late_30 + late_60 + late_90,
    }
    return pd.DataFrame([row])
