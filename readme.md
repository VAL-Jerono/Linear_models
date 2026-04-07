# Loan Analysis & Credit Risk Modelling — CRISP-DM

> **Two-task machine learning project** applying the CRISP-DM data science lifecycle to a 45,000-record loan dataset.  
> **Task A:** Predict applicant income (regression) · **Task B:** Predict loan default (classification)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Dataset Description](#3-dataset-description)
4. [CRISP-DM Phases](#4-crisp-dm-phases)
   - [Phase 1 & 2 — Business & Data Understanding](#phase-1--2--business--data-understanding)
   - [Phase 3 — Data Preparation](#phase-3--data-preparation)
   - [Phase 4A — Regression Modelling](#phase-4a--regression-modelling-task-a)
   - [Phase 4B — Classification Modelling](#phase-4b--classification-modelling-task-b)
   - [Phase 4C — Credit Risk Analysis](#phase-4c--credit-risk-analysis)
   - [Phase 5 — Evaluation](#phase-5--evaluation)
5. [Results Summary](#5-results-summary)
6. [Key Insights](#6-key-insights)
7. [Recommendations & Next Steps](#7-recommendations--next-steps)
8. [Bonus: Salary Dataset Scaffold](#8-bonus-salary-dataset-scaffold)
9. [Dependencies & Setup](#9-dependencies--setup)
10. [Usage](#10-usage)

---

## 1. Project Overview

This project follows the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) framework to model credit risk at a hypothetical lending institution — **QuickFin Bank**. It solves two concrete business problems:

| Task | Type | Target Variable | Goal |
|------|------|----------------|------|
| **A** | Regression | `person_income` | Predict applicant income for fraud detection & income verification |
| **B** | Classification | `loan_status` | Predict whether an applicant will default or repay |

**Why both tasks matter:** Income prediction supports fraud detection (applicants sometimes misreport income), while default prediction directly impacts profitability and portfolio risk.

---

## 2. Repository Structure

```
.
├── analysis.ipynb               # Main Jupyter notebook (Python, 44 cells)
├── Loan_Analysis_CRISP_DM.pptx  # Executive presentation (10 slides)
├── loan_data.csv                # Primary dataset (45,000 records) — user-supplied
├── salary_data.csv              # Secondary dataset (optional) — user-supplied
└── README.md
```

> **Note:** The raw CSV files are not included in this repository and must be supplied by the user. See [Usage](#10-usage) for details.

---

## 3. Dataset Description

### Loan Dataset (`loan_data.csv`)

| Property | Value |
|----------|-------|
| Records | 45,000 |
| Features | 14 (7 numeric, 7 categorical) |
| Missing values | None — no imputation required |
| Class balance (loan_status) | ~78% Repaid / ~22% Default |

**Columns:**

| Column | Type | Description |
|--------|------|-------------|
| `person_age` | Numeric | Applicant age (years) |
| `person_gender` | Categorical | Gender |
| `person_education` | Categorical | Highest education level (ordinal: High School → Doctorate) |
| `person_income` | Numeric | Annual income in USD — **regression target** |
| `person_emp_exp` | Numeric | Years of employment experience |
| `person_home_ownership` | Categorical | RENT / OWN / MORTGAGE / OTHER |
| `loan_amnt` | Numeric | Loan amount requested (USD) |
| `loan_intent` | Categorical | Purpose of the loan |
| `loan_int_rate` | Numeric | Interest rate (%) |
| `loan_percent_income` | Numeric | Loan amount as % of income |
| `cb_person_cred_hist_length` | Numeric | Credit history length (years) |
| `credit_score` | Numeric | Applicant credit score |
| `previous_loan_defaults_on_file` | Categorical | Yes / No |
| `loan_status` | Binary | 0 = Default, 1 = Repaid — **classification target** |

**Notable data quality issues found during EDA:**
- `person_age` contains values > 100 (data entry errors) — capped at 100.
- `person_income` has extreme outliers — capped at the 99th percentile.
- `person_emp_exp` also capped at 99th percentile.

---

## 4. CRISP-DM Phases

### Phase 1 & 2 — Business & Data Understanding

**Business questions addressed:**
- What demographic and financial factors drive income?
- Which borrowers are most likely to default?
- How does credit score interact with default risk?

**Success criteria set upfront:**
- Task A (Regression): Adjusted R² > 0.60 on the held-out test set
- Task B (Classification): Accuracy > 85%, AUC-ROC ≈ 1.0

**Key EDA findings:**
- Income is heavily right-skewed (skewness ≈ 1.4, high kurtosis) — the mean (~$80k) is not representative of the typical applicant.
- `person_age`, `person_emp_exp`, and `cb_person_cred_hist_length` are strongly correlated (financial maturity co-moves), raising multicollinearity risk.
- Higher income applicants carry proportionally lower loan-to-income ratios — a key signal of lower default risk.
- Home ownership type acts as an indirect wealth proxy: MORTGAGE/OWN holders earn significantly more than renters.
- `credit_score` has weak correlations with most other variables, suggesting standalone predictive power.

---

### Phase 3 — Data Preparation

The following preprocessing steps are applied in the notebook, in order:

**1. Outlier Removal**
```python
df = df[df['person_age'] <= 100]
df = df[df['person_income'] <= df['person_income'].quantile(0.99)]
df = df[df['person_emp_exp'] <= df['person_emp_exp'].quantile(0.99)]
# Removes ~1% of records as outliers
```

**2. Feature Engineering**

| New Feature | Formula | Rationale |
|-------------|---------|-----------|
| `debt_to_income` | `loan_amnt / person_income` | Measures financial stress |
| `income_per_exp_year` | `person_income / (person_emp_exp + 1)` | Earning efficiency by career stage |
| `credit_tier` | Binned from `credit_score` | Poor / Fair / Good / Very Good / Exceptional |
| `age_group` | Binned from `person_age` | Young Adult / Adult / Middle Age / Senior |

**3. Encoding**

| Variable | Method | Reasoning |
|----------|--------|-----------|
| `person_gender`, `previous_loan_defaults_on_file` | Label Encoding (binary) | Only two values |
| `person_education` | Ordinal Encoding (0–4) | Natural order: High School → Doctorate |
| `person_home_ownership`, `loan_intent` | One-Hot Encoding (drop_first=True) | Nominal, no natural order |

**4. Train / Test Split**
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y_clf  # stratified for classification
)
# 80% train / 20% test
```

**5. Log Transformation of Income (Task A)**
```python
y_transformed = np.log1p(person_income)
# After modelling: back-transform via np.expm1(y_pred)
```
Applied after diagnosing right-skew (skewness ≈ 1.4). Reduces heteroscedasticity and normalises residuals — a prerequisite for valid linear regression inference.

**6. Assumption Checking (LINER)**

All five linear regression assumptions are tested systematically:

| Assumption | Test | Pre-Transform Result | Post-Transform Result |
|------------|------|---------------------|----------------------|
| Linearity | Residuals vs Fitted | ❌ Curved pattern | ✅ Random scatter |
| Independence | Study design | ✅ Assumed | ✅ Assumed |
| Normality | Shapiro-Wilk, Jarque-Bera, Q-Q plot | ❌ p < 0.05 | ✅ Approximately normal |
| Equal Variance | Breusch-Pagan, Scale-Location | ❌ Fanning pattern | ✅ Stable |
| No Multicollinearity | VIF scores | ✅ All VIF < 5 | ✅ All VIF < 5 |

**VIF Summary:**

| Variable | VIF | Status |
|----------|-----|--------|
| `person_age` | 3.8 | ✅ OK |
| `person_emp_exp` | 3.5 | ✅ OK |
| `loan_amnt` | 2.1 | ✅ OK |
| `loan_int_rate` | 1.7 | ✅ OK |
| `credit_score` | 1.2 | ✅ OK |

No VIF > 5 — no severe multicollinearity detected.

---

### Phase 4A — Regression Modelling (Task A)

**Objective:** Predict `log(person_income)`, back-transform predictions to USD.

**Models compared:**

| Model | Key Characteristic |
|-------|--------------------|
| Linear Regression | Interpretable baseline |
| Ridge Regression (L2) | Handles multicollinearity |
| Lasso Regression (L1) | Built-in feature selection |
| Elastic Net (L1 + L2) | Balance of Ridge and Lasso |
| Decision Tree | Captures non-linear interactions |
| Random Forest | Robust ensemble, typically top performer |

Ridge, Lasso, and Elastic Net are trained on `StandardScaler`-normalised features. Tree-based models use raw (unscaled) features.

**Stepwise AIC selection** (`MASS::stepAIC` in the R version; manual in Python) was also applied to prune non-significant predictors and reduce AIC.

---

### Phase 4B — Classification Modelling (Task B)

**Objective:** Predict `loan_status` (0 = Default, 1 = Repaid).

**Why AUC-ROC over accuracy?**  
The dataset has a 78/22 class imbalance. A naïve "always predict Repaid" classifier would achieve 78% accuracy while being completely useless. AUC-ROC measures the model's ability to distinguish between classes at all decision thresholds.

**Models compared:**

| Model | Why Included |
|-------|-------------|
| Logistic Regression | Interpretable baseline; coefficients readable as log-odds |
| Decision Tree | Captures feature interactions, non-linear boundaries |
| Random Forest | Robust ensemble, typically top performer |
| Gradient Boosting | Often best out-of-box; highest sensitivity to defaults |

Logistic Regression is trained on scaled features; tree-based models use raw features.

**Preventing data leakage:** `person_income` is explicitly excluded from the classification feature set, since income is the regression target — including it would inflate classification metrics unrealistically.

---

### Phase 4C — Credit Risk Analysis

Deep-dive analysis into the relationship between income, credit score, and loan defaults, including:

- **Income vs default rate:** Boxplots showing defaulters have lower median income.
- **Credit score vs default rate:** Defaulters have meaningfully lower credit scores.
- **Default rate by education level:** Higher education correlates with lower default rates.
- **Default rate by home ownership:** Renters default at the highest rate — a strong standalone risk signal.

---

### Phase 5 — Evaluation

**5-Fold Cross-Validation** is run on all models to verify that test-set performance is not a result of a favourable random split. Results are reported as `mean ± std` of R² (regression) and AUC-ROC (classification).

**Model selection criteria:**

| Criterion | Weight | Notes |
|-----------|--------|-------|
| Performance | Primary | R² / AUC-ROC on held-out test set |
| Interpretability | Secondary | Can results be explained to a credit committee? |
| Robustness | Tertiary | Low cross-validation variance |

---

## 5. Results Summary

### Task A — Income Prediction (Regression)

| Metric | Raw Linear Model | Log-Transformed Best Model |
|--------|-----------------|---------------------------|
| Adjusted R² | 0.29 | **0.766** |
| RMSE (back-transformed) | — | ~$49,141 |
| MAE (back-transformed) | — | ~$16,591 |
| Assumptions met? | ❌ All 4 violated | ✅ Met post-transformation |
| R² improvement | — | +0.476 (+164%) |

**Winner:** Random Forest (highest R²) for fraud-detection use cases; Linear Regression for interpretability where coefficient-level explanation is needed.

### Task B — Default Prediction (Classification)

| Metric | Logistic Regression | Best Model (Gradient Boosting) |
|--------|--------------------|---------------------------------|
| Accuracy | 89.8% | Highest on test set |
| Cohen's Kappa | 0.70 | Substantial agreement |
| AUC-ROC | ≈ 1.0 | Near-perfect class separation |
| Sensitivity | High | Defaults correctly identified |

**Winner:** Gradient Boosting (best AUC-ROC) for risk-critical default detection; Logistic Regression where regulatory interpretability (log-odds coefficients) is required.

---

## 6. Key Insights

1. **Income inequality is structural.** The distribution is so right-skewed that the mean ($80k) is misleading without log transformation. Any income model built on raw values will violate all four linear regression assumptions.

2. **Log transformation is non-negotiable.** Applying `log(income)` raised R² from 0.29 → 0.77 — a 164% improvement — and corrected linearity, normality, and homoscedasticity simultaneously.

3. **Financial maturity co-moves.** Age, employment experience, and credit history length are strongly correlated. Log transformation untangles their individual effects on income prediction.

4. **Loan-to-income ratio is the clearest financial stress signal.** Used by both models as a top predictor; high ratios correlate with both lower income estimates and higher default probability.

5. **Home ownership is a reliable wealth proxy.** MORTGAGE/OWN holders default at significantly lower rates than renters — one of the strongest single categorical predictors.

6. **Previous loan defaults are the dominant classification signal.** Applicants with a prior default on file show dramatically elevated default rates regardless of current financial profile.

7. **Logistic regression coefficients are auditable.** Results can be expressed as odds ratios and presented to non-technical credit officers or regulators — a key advantage over black-box ensemble models.

---

## 7. Recommendations & Next Steps

### Immediate

- **HC3 robust standard errors:** Address residual heteroscedasticity that persists after log transformation in edge cases, without requiring perfect homoscedasticity.
- **SMOTE oversampling:** Balance the 78/22 class ratio in the training set to improve recall on the minority (Default) class.

### Short-Term

- **Ensemble benchmark:** Train Random Forest and XGBoost on the same feature set and compare RMSE and AUC against the current models.
- **k-Fold cross-validation (k=10):** Reduce dependency on the single 80/20 random split and produce more reliable generalisation estimates.

### Long-Term

- **Enrich with salary data:** Integrate the tech-sector salary dataset (see Section 8) to add richer income-prediction features across industries and roles.
- **Production deployment:**
  - Expose the logistic regression model via a **Plumber API** (R) or **FastAPI** (Python) for real-time scoring.
  - Build a **Shiny** (R) or **Streamlit** (Python) dashboard for credit officers to explore applicant profiles interactively.
- **Monitoring:** Implement model drift detection to retrain when population income distributions shift over time.

---

## 8. Bonus: Salary Dataset Scaffold

The notebook includes a second project scaffold (Cells 42–44) for analysing a tech-sector salary dataset (`salary_data.csv`). This project is structurally identical to the loan project:

| Task | Target | Goal |
|------|--------|------|
| Regression | `salary_in_usd` | Predict individual tech salaries |
| Classification | Salary bucket (Above/Below average) | Classify compensation competitiveness |

**Required columns for `salary_data.csv`:**
`work_year`, `experience_level`, `employment_type`, `job_title`, `salary`, `salary_currency`, `salary_in_usd`, `employee_residence`, `remote_ratio`, `company_location`, `company_size`

The salary analysis cells are gated behind a `SALARY_LOADED` flag and will print a clear warning if the file is missing, so running the loan analysis cells is unaffected.

---

## 9. Dependencies & Setup

### Python (Jupyter Notebook)

```bash
pip install pandas numpy matplotlib seaborn scipy statsmodels scikit-learn
```

| Library | Purpose |
|---------|---------|
| `pandas`, `numpy` | Data wrangling |
| `matplotlib`, `seaborn` | Visualisation |
| `scipy`, `statsmodels` | Statistical tests, VIF, Breusch-Pagan |
| `sklearn` | Preprocessing, models, metrics |

**Key sklearn imports used:**
- `train_test_split`, `cross_val_score`, `StratifiedKFold`
- `LabelEncoder`, `OneHotEncoder`, `StandardScaler`
- `LinearRegression`, `Ridge`, `Lasso`, `ElasticNet`
- `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`, `GradientBoostingClassifier`
- `mean_squared_error`, `r2_score`, `roc_auc_score`, `confusion_matrix`, `classification_report`

### R (Presentation / Reference Implementation)

The `.pptx` slides reference an R implementation using:

```r
install.packages(c("caret", "MASS", "pROC", "car", "ggplot2"))
```

| Package | Purpose |
|---------|---------|
| `caret` | Train/test split, model training |
| `MASS` | `stepAIC` for stepwise variable selection |
| `car` | `vif()` for multicollinearity, `ncvTest()` for heteroscedasticity |
| `pROC` | ROC curve plotting and AUC calculation |
| `ggplot2` | Publication-quality visualisations |

---

## 10. Usage

### 1. Place data files

Put `loan_data.csv` (and optionally `salary_data.csv`) in the same directory as `analysis.ipynb`.

### 2. Launch the notebook

```bash
jupyter notebook analysis.ipynb
```

### 3. Run cells in order

The notebook is designed to be run top-to-bottom. Cells are organised by CRISP-DM phase and are heavily commented. Each section prints diagnostic output and displays plots inline.

### 4. Interpreting outputs

- **Regression cells** report R², RMSE (log scale), and RMSE back-transformed to USD.
- **Classification cells** report Accuracy, AUC-ROC, Cohen's Kappa, and a full `classification_report`.
- **Diagnostic cells** print VIF tables, normality test statistics, and Breusch-Pagan results.
- The best model per task is automatically selected and announced with a trophy emoji (`🏆`).

### 5. Salary analysis (optional)

Place `salary_data.csv` in the same directory and run Cells 42–44. The notebook will auto-detect the file and proceed; if missing, it prints a clear warning and skips gracefully.

---

*Data Science · April 2026 · CRISP-DM Loan Analysis & Modelling*