# Credit Card Fraud Detection using LightGBM & Optuna

## Overview
This repository contains a Jupyter Notebook designed to detect fraudulent credit card transactions. The project addresses a binary classification problem with a highly imbalanced dataset (only 0.17% of transactions are fraudulent). It utilizes **LightGBM** for classification and **Optuna** for Bayesian hyperparameter optimization to maximize the detection of fraud cases while minimizing false positives.

## Dataset
The dataset used is the standard [Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud).
*   **Transactions:** 284,807
*   **Features:**
    *   `Time`: Seconds elapsed between this transaction and the first transaction in the dataset.
    *   `V1` to `V28`: Principal components obtained via PCA (anonymized).
    *   `Amount`: Transaction amount.
*   **Target:** `Class` (0 = Normal, 1 = Fraud).

**Class Imbalance:**
*   Non-Fraud (0): 284,315
*   Fraud (1): 492

## Tech Stack
*   **Language:** Python
*   **Data Manipulation:** Pandas, NumPy
*   **Visualization:** Matplotlib, Seaborn
*   **Machine Learning:** LightGBM, Scikit-learn
*   **Optimization:** Optuna
*   **Statistical Analysis:** Statsmodels (for VIF)

## Workflow

1.  **Data Loading & Inspection:**
    *   Checking data structure, types, and missing values.
    *   Statistical summary (`describe()`).
2.  **Exploratory Data Analysis (EDA):**
    *   **VIF Analysis:** Calculated Variance Inflation Factor to check for multicollinearity among features. 'Amount' showed a higher VIF (~11), while PCA features were low.
    *   **Class Distribution:** Confirmed the severe imbalance of the target variable.
3.  **Preprocessing:**
    *   Standard Train/Test split (80/20).
4.  **Hyperparameter Tuning (Optuna):**
    *   Objective: Maximize **F1-Score**.
    *   Tuned parameters: `num_leaves`, `max_depth`, `learning_rate`, `n_estimators`, `subsample`, `colsample_bytree`, `min_child_samples`, `reg_lambda`.
    *   Results visualized using Optuna's history, slice, and importance plots.
5.  **Model Training & Evaluation:**
    *   Training a LightGBM classifier using the best parameters found by Optuna.
    *   Evaluation metrics: Precision, Recall, F1-Score, and Confusion Matrix.

## Results

The optimized LightGBM model achieved strong performance on the test set:

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **0 (Non-Fraud)** | 1.00 | 1.00 | 1.00 | 56,862 |
| **1 (Fraud)** | **0.94** | **0.79** | **0.86** | 100 |

**Confusion Matrix:**
*   **True Negatives:** 56,857
*   **False Positives:** 5 (Extremely low false alarm rate)
*   **False Negatives:** 21
*   **True Positives:** 79

**Key Insights:**
*   **Feature Importance:** According to Optuna's analysis, `learning_rate`, `n_estimators`, and `reg_lambda` were the most critical hyperparameters for this specific problem.
*   **Performance:** The model successfully identified 79% of fraud cases (Recall) with very high confidence (94% Precision), effectively balancing the trade-off between missing fraud and annoying customers with false alarms.

## How to Run
1.  Ensure you have `creditcard.csv` in your working directory.
2.  Install dependencies:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn lightgbm optuna statsmodels
    ```
3.  Run the Jupyter Notebook.
