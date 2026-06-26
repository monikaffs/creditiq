"""Train models and save artifacts for the Streamlit app."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.config import (
    FEATURE_COLUMNS,
    FEATURE_LABELS,
    FEATURE_NAMES,
    METRICS,
    MODELS_DIR,
    SCALER,
    XGB_MODEL,
)
from src.data_pipeline import build_feature_matrix, preprocess_data


def evaluate_model(name: str, model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    return {
        "model": name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_test, y_prob)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
        },
        "pr_curve": {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
        },
    }


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = preprocess_data(save_cleaned=True)
    X, y = build_feature_matrix(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pos_count = (y_train == 1).sum()
    neg_count = (y_train == 0).sum()
    scale_pos_weight = neg_count / max(pos_count, 1)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=20,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=350,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        ),
    }

    comparison = []
    detailed = {}

    for name, model in models.items():
        if name == "XGBoost":
            model.fit(X_train, y_train)
            metrics = evaluate_model(name, model, X_test, y_test)
        else:
            model.fit(X_train_scaled, y_train)
            metrics = evaluate_model(name, model, X_test_scaled, y_test)

        comparison.append({k: metrics[k] for k in ["model", "accuracy", "precision", "recall", "f1", "auc_roc"]})
        detailed[name] = metrics

    xgb = models["XGBoost"]
    joblib.dump(xgb, XGB_MODEL)
    joblib.dump(scaler, SCALER)

    feature_importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": xgb.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    payload = {
        "comparison": comparison,
        "best_model": "XGBoost",
        "class_imbalance": {
            "default_rate": float(y.mean()),
            "scale_pos_weight": float(scale_pos_weight),
            "method": "scale_pos_weight in XGBoost; class_weight='balanced' for baseline models",
        },
        "dataset": {
            "rows": int(len(df)),
            "features": len(FEATURE_COLUMNS),
            "default_rate_pct": round(float(y.mean()) * 100, 2),
        },
        "xgboost": detailed["XGBoost"],
        "feature_importance": feature_importance.to_dict(orient="records"),
    }

    with open(METRICS, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    with open(FEATURE_NAMES, "w", encoding="utf-8") as f:
        json.dump(FEATURE_COLUMNS, f, indent=2)

    labels_path = MODELS_DIR / "feature_labels.json"
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(FEATURE_LABELS, f, indent=2)

    print("Training complete.")
    print(pd.DataFrame(comparison).to_string(index=False))
    print(f"\nSaved model to {XGB_MODEL}")


if __name__ == "__main__":
    main()
