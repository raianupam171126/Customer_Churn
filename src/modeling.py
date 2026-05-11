"""
Modeling Module
===============
Trains XGBoost and LightGBM classifiers with grid search hyperparameter
tuning and early stopping on PR-AUC.

Both models handle class imbalance via scale_pos_weight.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score


RANDOM_STATE = 42


# ──────────────────────────────────────────────
# Baseline
# ──────────────────────────────────────────────

def train_baseline(X_train, y_train):
    """Train a Logistic Regression baseline."""
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model


# ──────────────────────────────────────────────
# XGBoost
# ──────────────────────────────────────────────

DEFAULT_XGB_GRID = {
    "max_depth": [4, 6],
    "min_child_weight": [1, 5],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
    "gamma": [0, 1],
    "learning_rate": [0.03, 0.05],
}


def train_xgboost(
    X_train, y_train, X_val, y_val,
    param_grid: dict = None,
    n_estimators_search: int = 1000,
    n_estimators_final: int = 5000,
    early_stopping_rounds: int = 150,
) -> tuple:
    """
    Grid search + train final XGBoost model.

    Returns
    -------
    tuple
        (final_model, best_params, best_score)
    """
    if param_grid is None:
        param_grid = DEFAULT_XGB_GRID

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    # Grid search
    best_score = 0
    best_params = None

    for params in ParameterGrid(param_grid):
        model = xgb.XGBClassifier(
            n_estimators=n_estimators_search,
            objective="binary:logistic",
            eval_metric="aucpr",
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            tree_method="hist",
            n_jobs=-1,
            **params,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=0,
        )
        val_probs = model.predict_proba(X_val)[:, 1]
        score = average_precision_score(y_val, val_probs)

        if score > best_score:
            best_score = score
            best_params = params

    print(f"XGB Best PR-AUC: {best_score:.4f}")
    print(f"XGB Best Params: {best_params}")

    # Train final model with more estimators
    final_model = xgb.XGBClassifier(
        n_estimators=n_estimators_final,
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        tree_method="hist",
        n_jobs=-1,
        **best_params,
    )
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=0,
    )

    return final_model, best_params, best_score


# ──────────────────────────────────────────────
# LightGBM
# ──────────────────────────────────────────────

DEFAULT_LGB_GRID = {
    "num_leaves": [31, 64],
    "min_child_samples": [30, 50],
    "max_depth": [-1, 6],
    "learning_rate": [0.03, 0.05],
    "subsample": [0.75, 0.85],
    "colsample_bytree": [0.75, 0.85],
    "reg_alpha": [0.0, 1.0],
    "reg_lambda": [0.0, 1.0],
}


def train_lightgbm(
    X_train, y_train, X_val, y_val,
    param_grid: dict = None,
    n_estimators_search: int = 3000,
    n_estimators_final: int = 5000,
    early_stopping_rounds: int = 150,
) -> tuple:
    """
    Grid search + train final LightGBM model.

    Returns
    -------
    tuple
        (final_model, best_params, best_score)
    """
    if param_grid is None:
        param_grid = DEFAULT_LGB_GRID

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    best_score = 0
    best_params = None

    for params in ParameterGrid(param_grid):
        model = lgb.LGBMClassifier(
            n_estimators=n_estimators_search,
            objective="binary",
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            verbose=-1,
            **params,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="average_precision",
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
        )
        val_probs = model.predict_proba(X_val)[:, 1]
        score = average_precision_score(y_val, val_probs)

        if score > best_score:
            best_score = score
            best_params = params

    print(f"LGB Best PR-AUC: {best_score:.4f}")
    print(f"LGB Best Params: {best_params}")

    # Train final model
    final_model = lgb.LGBMClassifier(
        n_estimators=n_estimators_final,
        objective="binary",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        verbose=-1,
        **best_params,
    )
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
    )

    return final_model, best_params, best_score


# ──────────────────────────────────────────────
# Cross-Validation
# ──────────────────────────────────────────────

def cross_validate_model(
    model_cls,
    model_params: dict,
    X_train, y_train,
    n_splits: int = 5,
) -> list:
    """
    Stratified K-fold cross-validation returning PR-AUC per fold.

    Parameters
    ----------
    model_cls : class
        XGBClassifier or LGBMClassifier.
    model_params : dict
        Parameters to pass to the model constructor.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = []

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X_train, y_train)):
        X_tr = X_train.iloc[tr_idx]
        X_va = X_train.iloc[va_idx]
        y_tr = y_train.iloc[tr_idx]
        y_va = y_train.iloc[va_idx]

        model = model_cls(**model_params)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=0)

        probs = model.predict_proba(X_va)[:, 1]
        score = average_precision_score(y_va, probs)
        cv_scores.append(score)
        print(f"  Fold {fold+1}: PR-AUC = {score:.4f}")

    print(f"  Mean: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    return cv_scores
