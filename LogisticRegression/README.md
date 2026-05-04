# Customer Churn Prediction — Logistic Regression

> *A complete teaching notebook. From raw data to confident interpretation — no step skipped, no decision left unexplained.*

---

## What This Is

A fully annotated R notebook that teaches logistic regression the way it should be taught: through a real business problem, with every modelling decision explained before any code is written.

The dataset is a telecom company's customer records. The question is simple:
**which customers are about to leave — and why?**

That question makes logistic regression not just a statistical exercise but a revenue problem. The notebook keeps that tension alive throughout.

---

## The Problem With Most Tutorials

Most introductions to logistic regression hand you `glm()` in the first five minutes. You get output. You have no idea what the output means, why the data needed to be prepared the way it was, or what would have gone wrong if you had skipped any of the setup steps.

This notebook does the opposite. It earns each step.

---

## What You Will Learn

Work through this notebook once and you will be able to answer all of the following — not by looking them up, but because you watched each issue arise and watched it get solved.

**The model itself**
- Why logistic regression exists at all — what breaks when you use linear regression on a binary outcome
- What the sigmoid function does and why it guarantees a valid probability
- What log-odds are, and why the model operates in that space instead of probability space
- What IRLS is and how `glm()` actually fits the model

**Data preparation**
- Why `churn` must be cast to a factor before any modelling — and what R does silently if you forget
- What point-biserial correlation is and how to use it for feature selection
- What multicollinearity is, how to detect it with a heatmap, and which predictor to drop when two tell the same story
- Why z-score normalisation is not optional — and what "data leakage" means and costs you
- Why the scaler must be fitted on training data only, then applied to both sets

**Evaluation**
- Why accuracy is the wrong headline metric on imbalanced data
- What log-loss is, what it penalises, and what `0.6931` means as a baseline
- What AUC-ROC measures and why it is threshold-independent
- What the four cells of a confusion matrix represent in business terms (a missed churner is not the same cost as a false alarm)
- When to favour Recall over Precision — and how to shift that tradeoff on the ROC curve

**Interpretation**
- How to convert log-odds coefficients into odds ratios
- How to read a coefficient forest plot and what it means when a confidence interval crosses zero
- How to write a sentence about a model result that a non-statistician will believe

---

## Notebook Structure

| Step | What Happens | The Lesson |
|------|-------------|------------|
| 1 | Load libraries | All dependencies declared upfront — prevents mid-analysis surprises |
| 2 | EDA — structure, summaries, missing values, class balance | Never touch a model before you understand the data |
| 3 | Cast `churn` to factor | Without this, `glm()` fits linear regression; predictions escape [0,1] |
| 4 | Feature selection | Point-biserial correlation threshold; multicollinearity heatmap; drop log-transform duplicates |
| 5 | Normalisation | Stratified 80/20 split first; scaler fitted on train only; verified post-scaling |
| 6 | Train the model | `glm(family = binomial(link = "logit"))`; raw summary; odds ratios; forest plot |
| 7 | Evaluate | Log-loss, AUC-ROC, confusion matrix heatmap, full metrics table |
| 8 | Final summary | Consolidated printout of data, preprocessing, and all test-set metrics |

A concepts recap table and five suggested next steps follow the main pipeline.

---

## Requirements

**R packages** — install once, then the notebook loads them all:

```r
install.packages(c(
  "tidyverse",    # dplyr, ggplot2, readr, forcats
  "corrplot",     # correlation heatmap
  "scales",       # percent(), comma(), dollar() axis labels
  "caret",        # createDataPartition(), preProcess(), confusionMatrix()
  "pROC",         # roc(), auc()
  "MLmetrics",    # LogLoss()
  "broom",        # tidy() — clean glm() output
  "knitr",        # kable()
  "kableExtra"    # table styling
))
```

**Data file** — place `ChurnData.csv` in your working directory before knitting.

```r
getwd()          # check where R is looking
setwd("your/path/here")   # or use the Files pane in RStudio
```

---

## Rendering the Notebook

```r
rmarkdown::render("Churn_LogisticRegression_Teaching.Rmd")
```

Output: a self-contained HTML file with a floating table of contents, numbered sections, and all code visible (`code_folding: show`). Open it in any browser.

---

## A Note on the `if / else` Pattern in R

One thing this notebook will teach you implicitly: R's parser treats a brace-less `if` body as a complete expression the moment it hits a newline.

```r
# This breaks — R considers the if finished before it sees else
if (x < 0.4) cat("Good")
else          cat("Poor")   # Error: unexpected 'else'

# This works — the closing } tells R more is coming
if (x < 0.4) {
  cat("Good")
} else {
  cat("Poor")
}
```

Always use braces in multi-branch `if / else if / else` chains. The notebook follows this convention throughout.

---

## Concepts at a Glance

| Concept | R tool used | Why it matters |
|---------|-------------|----------------|
| Binary outcome → factor | `factor(churn, levels = c(0,1))` | Triggers binomial family in `glm()` |
| Sigmoid function | Built into `glm(family = binomial)` | Maps log-odds to valid probability in (0, 1) |
| Feature–outcome correlation | `cor()` + threshold `\|r\| > 0.10` | Removes noise, focuses the model |
| Multicollinearity detection | `corrplot()` heatmap | Identifies redundant predictors |
| Z-score normalisation | `caret::preProcess("center", "scale")` | Ensures IRLS converges reliably |
| No data leakage | Scaler fitted on train only | Honest evaluation metrics |
| Log-loss | `MLmetrics::LogLoss()` | What the model actually minimises |
| AUC-ROC | `pROC::roc()` + `auc()` | Threshold-independent ranking quality |
| Confusion matrix | `caret::confusionMatrix()` | TP / TN / FP / FN and all derived metrics |
| Odds ratios | `exp(coef(model))` | Interpretable effect sizes for each predictor |

---

## Suggested Next Steps

The notebook ends with working code stubs for five extensions:

1. **Class weights** — penalise missed churners more heavily than false alarms
2. **Threshold optimisation** — sweep the ROC curve to find the cutoff that fits the business cost structure
3. **Regularised logistic regression** — LASSO / Ridge via `glmnet`, with built-in feature selection
4. **Stepwise selection by AIC** — let `step()` prune the model greedily
5. **Non-linear comparison** — `randomForest` and `xgboost` to test whether the linear assumptions hold

---

## Who This Is For

- Students encountering logistic regression for the first time who want to understand the *why*, not just the *how*
- Practitioners who have used `glm()` but never felt confident interpreting the output
- Instructors who want a ready-made, heavily commented classroom notebook

By the end, the pipeline you have built — correlation-filtered features, leak-free normalisation, properly cast outcome, rigorous multi-metric evaluation — transfers directly to any binary classification problem you will ever face.

---

*Built with R · tidyverse · caret · pROC · MLmetrics · broom*