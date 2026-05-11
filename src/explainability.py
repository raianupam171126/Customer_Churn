"""
Explainability Module
=====================
SHAP-based model explainability for tree-based churn models.

Provides:
- Global feature importance (bar plot)
- Feature impact direction (beeswarm plot)
- Individual prediction explanations (waterfall/force plots)
"""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt


def compute_shap_values(model, X: pd.DataFrame) -> tuple:
    """
    Compute SHAP values using TreeExplainer.

    Parameters
    ----------
    model : fitted tree-based model
        XGBClassifier, LGBMClassifier, etc.
    X : pd.DataFrame
        Feature matrix.

    Returns
    -------
    tuple
        (explainer, shap_values)
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return explainer, shap_values


def plot_global_importance(shap_values, X: pd.DataFrame, max_display: int = 20):
    """
    Plot SHAP global feature importance (bar chart).

    Shows the mean absolute SHAP value per feature —
    which features matter most overall.
    """
    shap.summary_plot(shap_values, X, plot_type="bar", max_display=max_display)


def plot_beeswarm(shap_values, X: pd.DataFrame, max_display: int = 20):
    """
    Plot SHAP beeswarm (detailed impact direction).

    Shows how each feature value (high/low) pushes the prediction
    toward churn or retention.
    """
    shap.summary_plot(shap_values, X, max_display=max_display)


def explain_single_prediction(
    explainer,
    shap_values,
    X: pd.DataFrame,
    idx: int,
):
    """
    Explain a single customer's churn prediction.

    Parameters
    ----------
    idx : int
        Row index in X to explain.
    """
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[idx],
            base_values=explainer.expected_value,
            data=X.iloc[idx],
            feature_names=X.columns.tolist(),
        )
    )


def get_top_shap_features(
    shap_values, feature_names: list, top_n: int = 15
) -> pd.DataFrame:
    """
    Return top features ranked by mean |SHAP value|.

    Returns
    -------
    pd.DataFrame
        Columns: feature, mean_abs_shap
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False)

    return importance.head(top_n).reset_index(drop=True)
