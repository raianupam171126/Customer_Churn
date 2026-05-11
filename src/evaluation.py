"""
Evaluation Module
=================
Model evaluation utilities: metrics computation, precision-recall curves,
threshold optimization, and decile lift analysis.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, precision_recall_curve,
    classification_report,
)


# ──────────────────────────────────────────────
# Core Metrics
# ──────────────────────────────────────────────

def evaluate(model, X, y, threshold: float = 0.5) -> dict:
    """
    Compute Precision, Recall, F1, and PR-AUC.

    Parameters
    ----------
    model : fitted classifier
        Must have predict_proba method.
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        True labels.
    threshold : float
        Classification threshold.

    Returns
    -------
    dict
        {Precision, Recall, F1, PR-AUC}
    """
    probs = model.predict_proba(X)[:, 1]
    preds = (probs > threshold).astype(int)

    return {
        "Precision": round(precision_score(y, preds), 4),
        "Recall": round(recall_score(y, preds), 4),
        "F1": round(f1_score(y, preds), 4),
        "PR-AUC": round(average_precision_score(y, probs), 4),
    }


def print_classification_report(model, X, y, threshold: float = 0.5):
    """Print sklearn classification report."""
    probs = model.predict_proba(X)[:, 1]
    preds = (probs > threshold).astype(int)
    print(classification_report(y, preds))


# ──────────────────────────────────────────────
# Threshold Optimization
# ──────────────────────────────────────────────

def optimize_threshold_f1(model, X_val, y_val) -> dict:
    """
    Find the threshold that maximizes F1 score.

    Returns
    -------
    dict
        {threshold, precision, recall, f1}
    """
    probs = model.predict_proba(X_val)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_val, probs)

    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
    best_idx = np.argmax(f1_scores)

    return {
        "threshold": round(float(thresholds[best_idx]), 3),
        "precision": round(float(precision[best_idx]), 4),
        "recall": round(float(recall[best_idx]), 4),
        "f1": round(float(f1_scores[best_idx]), 4),
    }


def optimize_threshold_recall(
    model, X_val, y_val, min_precision: float = 0.40
) -> dict:
    """
    Find the threshold that maximizes recall while keeping
    precision above min_precision.

    Parameters
    ----------
    min_precision : float
        Minimum acceptable precision (default 0.40).

    Returns
    -------
    dict
        {threshold, precision, recall, f1}
    """
    probs = model.predict_proba(X_val)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_val, probs)

    valid_idx = np.where(precision[:-1] >= min_precision)[0]

    if len(valid_idx) == 0:
        print(f"No threshold achieves precision >= {min_precision}")
        return optimize_threshold_f1(model, X_val, y_val)

    best_idx = valid_idx[np.argmax(recall[valid_idx])]

    return {
        "threshold": round(float(thresholds[best_idx]), 3),
        "precision": round(float(precision[best_idx]), 4),
        "recall": round(float(recall[best_idx]), 4),
        "f1": round(float(2 * precision[best_idx] * recall[best_idx]
                          / (precision[best_idx] + recall[best_idx] + 1e-6)), 4),
    }


# ──────────────────────────────────────────────
# Decile Lift Analysis
# ──────────────────────────────────────────────

def decile_lift_table(model, X, y) -> pd.DataFrame:
    """
    Create a decile lift table showing churn rate per probability decile.

    The top decile should capture a disproportionate share of churners —
    this directly translates to campaign targeting efficiency.

    Returns
    -------
    pd.DataFrame
        Columns: decile, count, churn_count, churn_rate, cumulative_churn_rate, lift
    """
    probs = model.predict_proba(X)[:, 1]
    df = pd.DataFrame({"prob": probs, "target": y.values})

    df["decile"] = pd.qcut(df["prob"], 10, labels=False, duplicates="drop")

    overall_rate = df["target"].mean()

    lift = (
        df.groupby("decile")
        .agg(
            count=("target", "count"),
            churn_count=("target", "sum"),
            churn_rate=("target", "mean"),
        )
        .sort_index(ascending=False)
    )

    lift["lift"] = (lift["churn_rate"] / overall_rate).round(2)
    lift["cumulative_churn_rate"] = (
        lift["churn_count"].cumsum() / lift["count"].cumsum()
    ).round(4)

    return lift.reset_index()


def cumulative_gains(model, X, y) -> pd.DataFrame:
    """
    Compute cumulative gains: what % of all churners are captured
    by targeting the top N% of customers ranked by churn probability.
    """
    probs = model.predict_proba(X)[:, 1]
    df = pd.DataFrame({"prob": probs, "target": y.values})
    df = df.sort_values("prob", ascending=False)

    df["cum_churners"] = df["target"].cumsum()
    df["cum_pct_churners"] = df["cum_churners"] / df["target"].sum()
    df["pct_population"] = np.arange(1, len(df) + 1) / len(df)

    return df[["pct_population", "cum_pct_churners"]]


# ──────────────────────────────────────────────
# Feature Importance
# ──────────────────────────────────────────────

def get_feature_importance(model, feature_names: list, top_n: int = 20) -> pd.DataFrame:
    """Extract and rank feature importances from a tree-based model."""
    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    return importance.head(top_n).reset_index(drop=True)
