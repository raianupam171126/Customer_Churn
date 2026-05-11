# 🔄 Customer Churn Prediction

**End-to-end churn prediction pipeline using XGBoost and LightGBM with target encoding, threshold optimization, SHAP explainability, and decile lift analysis — built for imbalanced B2B/B2C datasets.**

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Business Problem

Customer churn is one of the most expensive problems in subscription and recurring-revenue businesses. Acquiring a new customer costs 5–7x more than retaining an existing one. Yet most churn happens silently — by the time a customer formally cancels, the window for intervention has already closed.

**This project builds a predictive model that:**
- Identifies customers likely to churn **before** they leave
- Ranks customers by churn probability so retention teams can **prioritize outreach**
- Explains **why** each customer is at risk using SHAP values (not just "who")
- Optimizes the prediction threshold to **maximize recall** while maintaining acceptable precision
- Produces a **decile lift chart** showing how much better the model is than random targeting

---

## Pipeline Overview

```
Step 1:   Data Loading & Merging (features + labels join)
Step 2:   Data Validation (shape, types, missing values, target distribution)
Step 3:   Exploratory Data Analysis
          ├── Target class distribution
          ├── Correlation heatmap
          └── Feature distributions by churn status (boxplots)
Step 4:   Feature Engineering
          ├── Drop columns with >90% missing values
          ├── Target encoding for categorical features (5-fold CV)
          └── Mean imputation for remaining numerical NaNs
Step 5:   Train / Validation / Test Split (60/20/20, stratified)
Step 6:   Model Training
          ├── Logistic Regression (baseline)
          ├── XGBoost (grid search over depth, subsample, learning rate)
          └── LightGBM (grid search over leaves, depth, regularization)
Step 7:   Cross-Validation (5-fold stratified, PR-AUC metric)
Step 8:   Model Evaluation (Precision, Recall, F1, PR-AUC)
Step 9:   Threshold Optimization (maximize F1 / maximize recall at precision ≥ 0.4)
Step 10:  SHAP Explainability (global feature importance + beeswarm)
Step 11:  Decile Lift Analysis (cumulative gains chart)
```

---

## Data

| Detail | Value |
|:---|:---|
| Source | Competition/real-world dataset (features + labels as separate CSVs) |
| Split | 60% train / 20% validation / 20% test (stratified) |
| Target | Binary churn label (imbalanced) |
| Features | Mix of numerical and categorical |
| Handling | Target encoding (CV-regularized), mean imputation, high-missing column drop |

---

## Key Techniques

### Target Encoding (Leak-Free)
Categorical features are encoded using 5-fold stratified cross-validation to prevent target leakage. Each fold's encoding is trained only on out-of-fold data, then validation and test sets use the full training set's encoding.

### Class Imbalance Handling
Churn datasets are inherently imbalanced. Both XGBoost and LightGBM use `scale_pos_weight` (ratio of negatives to positives) to up-weight the minority class during training.

### Grid Search Hyperparameter Tuning
Both models are tuned across key hyperparameters with early stopping on validation PR-AUC:

**XGBoost:** max_depth, min_child_weight, subsample, colsample_bytree, gamma, learning_rate

**LightGBM:** num_leaves, min_child_samples, max_depth, learning_rate, subsample, colsample_bytree, reg_alpha, reg_lambda

### Threshold Optimization
Default 0.5 threshold is rarely optimal for imbalanced data. Two strategies are evaluated:
- **Max F1** — best balance of precision and recall
- **Recall-focused** — maximize recall subject to precision ≥ 0.40 (when the cost of missing a churner is high)

### SHAP Explainability
TreeSHAP provides both global feature importance (bar plot) and per-feature impact direction (beeswarm plot), enabling the business to understand not just *who* will churn, but *why*.

### Decile Lift Analysis
Customers are ranked by predicted churn probability and grouped into deciles. The lift table shows how much more effective the model is at finding churners in the top deciles compared to random selection — directly translating to campaign ROI.

---

## Results Summary

### Model Comparison

| Model | PR-AUC (Val) | Precision | Recall | F1 |
|:---|:---|:---|:---|:---|
| Logistic Regression | Baseline | — | — | — |
| XGBoost | Tuned | Tuned | Tuned | Tuned |
| **LightGBM (Final)** | **Best** | **Optimized** | **Optimized** | **Optimized** |

> LightGBM selected as final model based on validation PR-AUC. Exact metrics depend on data — run the notebook to reproduce.

### Visualizations

- Target class distribution bar chart
- Correlation heatmap for numerical features
- Feature-by-churn boxplots (top 5 features)
- Precision-Recall curve with AUC
- SHAP global importance (bar) and beeswarm plots
- Decile lift table and cumulative gains

---

## Project Structure

```
customer-churn-prediction/
│
├── data/
│   ├── raw/                          # Place train.csv and train_churn_labels.csv here
│   └── processed/                    # Encoded features and predictions
│
├── notebooks/
│   └── Customer_Churn.ipynb          # Full end-to-end pipeline
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py              # Missing value handling, target encoding, splitting
│   ├── modeling.py                    # XGBoost, LightGBM training with grid search
│   ├── evaluation.py                 # Metrics, PR curve, threshold optimization, lift analysis
│   └── explainability.py             # SHAP wrapper for TreeExplainer
│
├── outputs/                          # Saved charts, SHAP plots, lift tables
├── requirements.txt
├── README.md
└── LICENSE
```

---

## How to Run

```bash
# Clone the repository
git clone https://github.com/raianupam171126/customer-churn-prediction.git
cd customer-churn-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Place your data files in data/raw/
# Then launch the notebook
jupyter notebook notebooks/Customer_Churn.ipynb
```

**Google Colab:**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/raianupam171126/customer-churn-prediction/blob/main/notebooks/Customer_Churn.ipynb)

---

## Tech Stack

- **Python 3.10** — pandas, NumPy
- **Gradient Boosting** — XGBoost, LightGBM
- **Baseline** — scikit-learn (LogisticRegression)
- **Encoding** — Target Encoding (StratifiedKFold CV-regularized)
- **Evaluation** — PR-AUC, Precision-Recall curves, threshold optimization
- **Explainability** — SHAP (TreeExplainer)
- **Visualization** — Matplotlib, Seaborn

---

## Limitations & Future Work

- **Feature engineering** — current pipeline uses raw features + target encoding; domain-specific features (tenure bins, usage trends, engagement velocity) could improve lift
- **No time-based validation** — uses random stratified split; real-world churn should use temporal splits (train on month M, predict M+1)
- **Single threshold** — production systems could use multiple thresholds for tiered interventions (high-risk → call, medium-risk → email, low-risk → automated)
- **No cost-sensitive learning** — a custom loss function incorporating actual retention costs and customer LTV would optimize business value directly
- **Model monitoring** — production deployment needs drift detection and periodic retraining as customer behavior evolves

---

## References

- Lundberg, S. & Lee, S.I. (2017). *A Unified Approach to Interpreting Model Predictions* (SHAP)
- Chen, T. & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*
- Ke, G. et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Connect

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://linkedin.com/in/YOUR-PROFILE)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/raianupam171126)
