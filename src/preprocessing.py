"""
Preprocessing Module
====================
Handles data loading, merging, missing value treatment, target encoding,
and train/validation/test splitting for churn prediction.

Pipeline:
    Load CSVs → Merge features + labels → Drop high-missing columns →
    Target-encode categoricals (CV-regularized) → Mean-impute numerics → Split
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold


RANDOM_STATE = 42


# ──────────────────────────────────────────────
# Data Loading & Merging
# ──────────────────────────────────────────────

def load_and_merge(
    features_path: str,
    labels_path: str,
    key_col: str = "customer_id",
    target_col: str = "target",
) -> pd.DataFrame:
    """
    Load feature and label CSVs, merge, and return a single DataFrame.

    Parameters
    ----------
    features_path : str
        Path to features CSV.
    labels_path : str
        Path to labels CSV.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with target column.
    """
    features = pd.read_csv(features_path)
    labels = pd.read_csv(labels_path)

    features.reset_index(inplace=True)
    labels.reset_index(inplace=True)

    features.rename(columns={"index": key_col}, inplace=True)
    labels.rename(columns={"index": target_col}, inplace=True)

    data = features.merge(labels, left_on=key_col, right_on=target_col, how="inner")

    # Clean up columns
    if target_col in data.columns and "Label" in data.columns:
        data.drop(columns=[target_col], inplace=True)
        data.rename(columns={"Label": target_col}, inplace=True)

    if key_col in data.columns:
        data.drop(columns=[key_col], inplace=True)

    return data


# ──────────────────────────────────────────────
# Missing Value Handling
# ──────────────────────────────────────────────

def drop_high_missing_columns(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    threshold: float = 0.9,
) -> tuple:
    """
    Drop columns with more than `threshold` fraction of missing values.

    Uses training set to determine which columns to drop,
    then applies the same drop to validation and test sets.
    """
    missing_pct = X_train.isnull().mean()
    drop_cols = missing_pct[missing_pct > threshold].index.tolist()

    X_train = X_train.drop(columns=drop_cols)
    X_val = X_val.drop(columns=drop_cols)
    X_test = X_test.drop(columns=drop_cols)

    print(f"Dropped {len(drop_cols)} columns with >{threshold*100:.0f}% missing: {drop_cols}")
    return X_train, X_val, X_test, drop_cols


def impute_numerical(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple:
    """Mean-impute numerical columns using training set statistics."""
    num_cols = X_train.select_dtypes(include=np.number).columns

    means = {}
    for col in num_cols:
        mean_val = X_train[col].mean()
        means[col] = mean_val
        X_train[col] = X_train[col].fillna(mean_val)
        X_val[col] = X_val[col].fillna(mean_val)
        X_test[col] = X_test[col].fillna(mean_val)

    return X_train, X_val, X_test, means


# ──────────────────────────────────────────────
# Target Encoding (Leak-Free)
# ──────────────────────────────────────────────

def target_encode(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    y: pd.Series,
    cols: list,
    n_splits: int = 5,
    random_state: int = RANDOM_STATE,
) -> tuple:
    """
    Target-encode categorical columns using stratified K-fold CV
    to prevent target leakage.

    Parameters
    ----------
    train, val, test : pd.DataFrame
        Feature DataFrames.
    y : pd.Series
        Training target.
    cols : list
        Categorical columns to encode.
    n_splits : int
        Number of CV folds.

    Returns
    -------
    tuple
        (train_encoded, val_encoded, test_encoded)
    """
    train_enc = train.copy()
    val_enc = val.copy()
    test_enc = test.copy()

    global_mean = y.mean()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for col in cols:
        train_enc[col] = np.nan

        # Out-of-fold encoding for training set
        for tr_idx, va_idx in skf.split(train, y):
            fold_means = y.iloc[tr_idx].groupby(train[col].iloc[tr_idx]).mean()
            train_enc[col].iloc[va_idx] = train[col].iloc[va_idx].map(fold_means)

        train_enc[col].fillna(global_mean, inplace=True)

        # Full training set encoding for val/test
        full_means = y.groupby(train[col]).mean()
        val_enc[col] = val[col].map(full_means).fillna(global_mean)
        test_enc[col] = test[col].map(full_means).fillna(global_mean)

    return train_enc, val_enc, test_enc


# ──────────────────────────────────────────────
# Train / Validation / Test Split
# ──────────────────────────────────────────────

def create_splits(
    data: pd.DataFrame,
    target_col: str = "target",
    test_size: float = 0.4,
    val_ratio: float = 0.5,
    random_state: int = RANDOM_STATE,
) -> tuple:
    """
    Stratified 60/20/20 split.

    Returns
    -------
    tuple
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    X = data.drop(columns=[target_col])
    y = data[target_col]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=val_ratio, stratify=y_temp, random_state=random_state
    )

    print(f"Train: {X_train.shape[0]:,}  Val: {X_val.shape[0]:,}  Test: {X_test.shape[0]:,}")
    print(f"Churn rate — Train: {y_train.mean():.3f}  Val: {y_val.mean():.3f}  Test: {y_test.mean():.3f}")

    return X_train, X_val, X_test, y_train, y_val, y_test
