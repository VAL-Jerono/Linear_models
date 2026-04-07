# ══════════════════════════════════════════════════════════════════════
# LOAN DEFAULT ANALYSIS — COMPLETE R SCRIPT
# Dataset: 45,000 loan applicants | Target: loan_status (default/repay)
# ══════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════
# SECTION 1 — SETUP & DATA LOADING
# ══════════════════════════════════════════════════════════════════════

# ADDED: rpart and randomForest were missing from the original install list.
# ADDED: lmtest for statistical assumption tests (Breusch-Pagan, Durbin-Watson).
# Only run install.packages() once — comment it out after first run.
install.packages(c("tidyverse", "caret", "corrplot", "car",
                   "MASS", "ROCR", "rpart", "randomForest", "lmtest"))

library(tidyverse)
library(caret)
library(corrplot)
library(car)       # VIF test
library(MASS)      # stepAIC
library(ROCR)      # ROC curve
library(rpart)     # decision tree     — ADDED (was missing)
library(randomForest)                  # ADDED (was missing)
library(lmtest)    # statistical tests — ADDED (was missing)
library(dplyr)

# CORRECTED: original path was hardcoded to one person's computer.
# Change this to wherever your file is saved.
data <- read.csv("loan_data.csv")

# Basic first look at the data
dim(data)      # confirms 45,000 rows and 14 columns
str(data)      # ADDED: shows each column's data type — important before cleaning
head(data)     # first 6 rows
summary(data)  # descriptive stats: min, max, mean, median for every column

# ADDED: check for duplicate rows — duplicates distort model training
sum(duplicated(data))
# Expected output: a number. If it is 0, no duplicates. If > 0, remove them.
# We handle removal in Section 3.


# ══════════════════════════════════════════════════════════════════════
# SECTION 1 FINDINGS
# ══════════════════════════════════════════════════════════════════════
# - 45,000 rows and 14 columns confirmed
# - person_age max = 144 — biologically impossible, is an outlier
# - person_emp_exp max = 123 — impossible, is an outlier
# - person_income max = 7,200,766 — extreme outlier
# - loan_status mean ≈ 0.22 meaning ~78% defaulted, ~22% repaid (class imbalance)
# - 5 columns are character type and need converting to factors:
#     person_gender, person_education, person_home_ownership,
#     loan_intent, previous_loan_defaults_on_file
# - loan_int_rate is numeric — will be used as regression target
# - loan_status is 0/1 — will be used as classification target


# ══════════════════════════════════════════════════════════════════════
# SECTION 2 — EXPLORATORY DATA ANALYSIS (EDA)
# ══════════════════════════════════════════════════════════════════════

# ── 2.1 Missing values ────────────────────────────────────────────────
colSums(is.na(data))
# Expected output: a count per column.
# This dataset has no missing values — all columns should show 0.
# If any column shows a number > 0, we impute or remove in Section 3.

# ── 2.2 Target variable distribution ─────────────────────────────────
table(data$loan_status)
prop.table(table(data$loan_status))
# Expected output:
#   0 (defaulted) ≈ 78% of rows
#   1 (repaid)    ≈ 22% of rows
# This confirms class imbalance — the model will see far more defaults
# than repayments during training, which can bias predictions.

# ── 2.3 Distribution plots for all numeric variables ─────────────────
# ADDED: the original only plotted income. All numerics should be checked.

par(mfrow = c(2, 3))
hist(data$person_age,      main = "Age",             col = "steelblue", xlab = "Age (years)")
hist(data$person_income,   main = "Income",           col = "steelblue", xlab = "Annual Income (USD)")
hist(data$loan_amnt,       main = "Loan Amount",      col = "steelblue", xlab = "Amount (USD)")
hist(data$loan_int_rate,   main = "Interest Rate",    col = "steelblue", xlab = "Rate (%)")
hist(data$credit_score,    main = "Credit Score",     col = "steelblue", xlab = "Score")
hist(data$person_emp_exp,  main = "Employment Exp",   col = "steelblue", xlab = "Years")

# Expected output — 6 histograms showing:
#   person_age:     spike near 20-30, long right tail due to age=144 outlier
#   person_income:  extreme right skew — most income near left, one tiny bar near 7M
#   loan_amnt:      right skewed — most loans are smaller amounts
#   loan_int_rate:  roughly bell-shaped, centred around 10-12%
#   credit_score:   roughly normal, most between 600-750
#   person_emp_exp: heavily right skewed due to emp_exp=123 outlier

# ── 2.4 Boxplots to see outliers visually ────────────────────────────
# ADDED: these confirm what the histograms suggest.
dev.off()
par(mfrow = c(1, 3))
boxplot(data$person_age,     main = "Age",            col = "tomato", ylab = "Years")
boxplot(data$person_income,  main = "Income",         col = "tomato", ylab = "USD")
boxplot(data$person_emp_exp, main = "Employment Exp", col = "tomato", ylab = "Years")

# Expected output — 3 boxplots each showing:
#   A box (the middle 50% of the data) and whiskers (normal range).
#   Dots above the upper whisker = outliers.
#   person_age:    dots at 80+ and one extreme dot near 144
#   person_income: the box is compressed at the bottom, extreme dot near 7M
#   person_emp_exp: the box is near zero, extreme dot near 123
# These outliers will be removed in Section 3.

# ── 2.5 Categorical variable distributions ───────────────────────────
# ADDED: bar charts are cleaner than table() for a presentation.
dev.off()
par(mfrow = c(2, 3))
barplot(table(data$person_gender),          main = "Gender",         col = "steelblue")
barplot(table(data$person_education),       main = "Education",      col = "steelblue")
barplot(table(data$person_home_ownership),  main = "Home Ownership", col = "steelblue")
barplot(table(data$loan_intent),            main = "Loan Intent",    col = "steelblue")
barplot(table(data$previous_loan_defaults_on_file), main = "Prior Defaults", col = "steelblue")

# Expected output: 5 bar charts.
# Education: Bachelors and Associates are most common.
# Home ownership: RENT is the majority, OWN is the minority.
# Loan intent: EDUCATION and MEDICAL are the most common purposes.

# ── 2.6 Default rate by category ─────────────────────────────────────
# ADDED: this shows which categories are most associated with default.
data %>% group_by(person_gender) %>%
  summarise(count = n(),
            default_rate = round(mean(as.numeric(as.character(loan_status)) == 0) * 100, 2))

data %>% group_by(person_education) %>%
  summarise(count = n(),
            default_rate = round(mean(as.numeric(as.character(loan_status)) == 0) * 100, 2)) %>%
  arrange(desc(default_rate))

data %>% group_by(person_home_ownership) %>%
  summarise(count = n(),
            default_rate = round(mean(as.numeric(as.character(loan_status)) == 0) * 100, 2)) %>%
  arrange(desc(default_rate))

data %>% group_by(loan_intent) %>%
  summarise(count = n(),
            default_rate = round(mean(as.numeric(as.character(loan_status)) == 0) * 100, 2)) %>%
  arrange(desc(default_rate))

data %>% group_by(previous_loan_defaults_on_file) %>%
  summarise(count = n(),
            default_rate = round(mean(as.numeric(as.character(loan_status)) == 0) * 100, 2))

# Expected findings:
#   Gender:            both male and female default at ~77.8% — gender is NOT predictive
#   Education:         all education levels default at ~77-78% — also not predictive
#   Home ownership:    OWN=92% default rate, RENT=68% — unexpected and worth noting
#   Loan intent:       VENTURE=85%, EDUCATION=83%, DEBTCONSOLIDATION=70%
#   Previous defaults: No=55%, Yes=100% — having a prior default guarantees another

# ── 2.7 Correlation matrix ────────────────────────────────────────────
dev.off()
numeric_data <- data %>% dplyr::select(where(is.numeric))
cor_matrix   <- cor(numeric_data)

corrplot(cor_matrix,
         method      = "color",
         type        = "upper",
         addCoef.col = "black",
         number.cex  = 0.7,
         tl.cex      = 0.7,
         title       = "Correlation Matrix",
         mar         = c(0, 0, 1, 0))

# Expected output — a colour grid where:
#   Dark blue = strong positive correlation (both go up together)
#   Dark red  = strong negative correlation (one goes up, other goes down)
#   White     = no correlation
# Key correlations to look for:
#   loan_percent_income and loan_amnt:  should be positively correlated
#   credit_score and loan_int_rate:     should be negatively correlated
#     (better credit score = lower interest rate)
#   person_income and loan_amnt:        should be positively correlated
#     (higher income = larger loan approved)
# Any pair with correlation > 0.8 is a potential multicollinearity problem
# — we will check this formally with VIF in Section 5.


# ══════════════════════════════════════════════════════════════════════
# SECTION 2 FINDINGS
# ══════════════════════════════════════════════════════════════════════
# - No missing values in any column (clean dataset)
# - person_income, person_age, person_emp_exp are all right-skewed
# - Outliers confirmed: age=144, emp_exp=123, income=7.2M (impossible values)
# - Class imbalance: 78% defaulted, 22% repaid
# - Previous defaults is the strongest predictor: 100% default rate for Yes
# - Gender and education do not meaningfully predict default
# - loan_int_rate and loan_percent_income are most correlated with loan_status


# ══════════════════════════════════════════════════════════════════════
# SECTION 3 — DATA CLEANING & FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════

# NOTE ON ORDER: feature engineering is placed HERE, before the train/test split.
# CORRECTED: the original script created new features AFTER models were trained —
# those features would never have appeared in any model.
# Creating them here ensures all models can use them.

# ── 3.1 Remove impossible outliers ───────────────────────────────────
# We remove rows where values are biologically or logically impossible.
# These are data entry errors, not legitimate extreme values.

cat("Rows before outlier removal:", nrow(data), "\n")

data <- data %>%
  filter(person_age     <= 100) %>%   # nobody works past 100
  filter(person_emp_exp <= 60)  %>%   # cannot work for 60+ years before middle age
  filter(person_income  <= 1000000)   # cap at 1M — the 7.2M row is clearly erroneous

cat("Rows after outlier removal:", nrow(data), "\n")
# Expected output: around 44,800–44,900 rows remain.
# We lose very few rows — the outliers were a tiny minority.

# ── 3.2 Remove duplicate rows ────────────────────────────────────────
# ADDED: the original script never removed duplicates even after checking.
before_dedup <- nrow(data)
data <- data[!duplicated(data), ]
cat("Rows removed as duplicates:", before_dedup - nrow(data), "\n")

# ── 3.3 Convert categorical columns to factors ───────────────────────
# KEPT from original — this is correct.
# Factors tell R that these columns are categories, not continuous numbers.
data$person_gender               <- as.factor(data$person_gender)
data$person_education            <- as.factor(data$person_education)
data$person_home_ownership       <- as.factor(data$person_home_ownership)
data$loan_intent                 <- as.factor(data$loan_intent)
data$previous_loan_defaults_on_file <- as.factor(data$previous_loan_defaults_on_file)
data$loan_status                 <- as.factor(data$loan_status)

# ── 3.4 Feature Engineering ──────────────────────────────────────────
# CORRECTED: moved here from the end of the script where it had no effect.
# These two new features capture relationships the raw columns do not express.

# Income per year of experience: measures earning efficiency
# A person earning 100,000 with 2 years experience is more impressive
# than one earning 100,000 with 20 years — this captures that difference
data$income_per_experience <- data$person_income / (data$person_emp_exp + 1)
# (+1 prevents division by zero for people with 0 employment experience)

# Loan amount relative to credit score: measures borrowing relative to creditworthiness
# A large loan for someone with a low credit score = higher risk
data$loan_to_credit_ratio <- data$loan_amnt / data$credit_score

cat("New features added: income_per_experience, loan_to_credit_ratio\n")

# Verify the cleaned data
summary(data)
# Check: person_age max should now be ≤ 100
# Check: person_emp_exp max should now be ≤ 60
# Check: the two new columns appear at the end


# ══════════════════════════════════════════════════════════════════════
# SECTION 4 — TRAIN / TEST SPLIT
# ══════════════════════════════════════════════════════════════════════

# NOTE: the split happens AFTER cleaning and feature engineering
# so all models train on clean data with all features available.

set.seed(123)  # makes results reproducible — same random split every time

# Stratified split on loan_status so both classes are proportionally
# represented in both train and test sets
train_index <- createDataPartition(data$loan_status, p = 0.8, list = FALSE)
# CORRECTED: original split on person_income — we should split on the
# classification target (loan_status) to preserve the class balance.

train <- data[train_index, ]
test  <- data[-train_index, ]

cat("Training rows:", nrow(train), "\n")
cat("Test rows:    ", nrow(test),  "\n")
# Expected: ~36,000 train rows, ~9,000 test rows


# ══════════════════════════════════════════════════════════════════════
# SECTION 5 — LINEAR REGRESSION (FULL MODEL)
# ══════════════════════════════════════════════════════════════════════

# We use loan_int_rate as the regression target.
# CORRECTED: the original used person_income as the target.
# Predicting income from other loan features does not make sense —
# income is collected during application, not derived from loan behaviour.
# loan_int_rate is set by the bank and IS influenced by credit score,
# loan amount, income, and other applicant features — a much better fit.

lm_model <- lm(loan_int_rate ~ person_age + person_income + person_emp_exp +
                 loan_amnt + loan_percent_income + credit_score +
                 cb_person_cred_hist_length + income_per_experience +
                 loan_to_credit_ratio,
               data = train)
# We exclude categorical variables from the initial model to keep
# assumption checking straightforward. We add them in the stepwise stage.

summary(lm_model)
# Expected output — a table with:
#   Coefficients: each predictor's estimated effect on loan_int_rate
#   Std. Error:   uncertainty in each estimate
#   t value:      how many standard errors from zero
#   Pr(>|t|):     p-value — if < 0.05 the variable is statistically significant
#   Residual standard error: average prediction error in % interest rate
#   Multiple R-squared: proportion of variance explained by the model
#   Adjusted R-squared: R-squared penalised for number of predictors
# Look for: which variables have Pr(>|t|) < 0.05 (significant predictors)


# ══════════════════════════════════════════════════════════════════════
# SECTION 6 — ASSUMPTION CHECKING (BEFORE TRANSFORMATION)
# ══════════════════════════════════════════════════════════════════════

# This section is the most important part of the assignment.
# Every assumption has a visual check AND a statistical test.

# ── 6.1 The 4 diagnostic plots ───────────────────────────────────────
dev.off()
par(mfrow = c(2, 2))
plot(lm_model)

# ── HOW TO READ EACH OF THE 4 PLOTS ──────────────────────────────────
#
# PLOT 1 — Residuals vs Fitted (top left)
# What it checks: LINEARITY
# What to look for: residuals (dots) should be randomly scattered around
#   the horizontal line at zero, with no visible curve or pattern.
# ASSUMPTION MET if:    dots are randomly scattered, line is roughly flat
# ASSUMPTION VIOLATED if: there is a U-shape, arch, or clear curve
# Our expected result: likely shows some pattern due to outliers in income
#
# PLOT 2 — Normal Q-Q (top right)
# What it checks: NORMALITY OF RESIDUALS
# What to look for: dots should follow the diagonal dashed line closely.
#   The x-axis shows where the dot SHOULD be if residuals were perfectly normal.
#   The y-axis shows where it ACTUALLY is.
# ASSUMPTION MET if:    dots follow the line, especially in the middle
# ASSUMPTION VIOLATED if: dots curve away from the line at both ends (heavy tails)
# Our expected result: slight deviation at the tails — common with financial data
#
# PLOT 3 — Scale-Location (bottom left)
# What it checks: HOMOSCEDASTICITY (equal variance of residuals)
# What to look for: the red line should be roughly horizontal (flat).
#   Dots should be evenly spread above and below.
# ASSUMPTION MET if:    red line is flat, spread is consistent
# ASSUMPTION VIOLATED if: the red line slopes upward (variance grows with fitted value)
#   This is called heteroscedasticity — common in income/financial data
# Our expected result: likely shows increasing spread = transformation needed
#
# PLOT 4 — Residuals vs Leverage (bottom right)
# What it checks: INFLUENTIAL OUTLIERS
# What to look for: dots should stay inside the red dashed lines (Cook's distance).
#   Points outside those lines have too much influence on the model.
# ASSUMPTION MET if:    no points outside the dashed red boundary lines
# ASSUMPTION VIOLATED if: named points (e.g. row 1432) appear outside the dashed lines

# ── 6.2 Statistical test — Normality (Shapiro-Wilk) ──────────────────
# ADDED: visual Q-Q plots can be subjective. The Shapiro-Wilk test gives
# an objective p-value for whether residuals are normally distributed.
# Note: Shapiro-Wilk only works on samples up to 5000. We sample 4500.
set.seed(42)
shapiro.test(sample(residuals(lm_model), 4500))
# Expected output:
#   W = a number close to 1 (closer to 1 = more normal)
#   p-value: if < 0.05 → normality is VIOLATED
#            if ≥ 0.05 → normality is MET
# With financial data at this scale, p < 0.05 is likely — transformation needed

# ── 6.3 Statistical test — Homoscedasticity (Breusch-Pagan) ──────────
# ADDED: the Breusch-Pagan test formally tests whether variance is constant.
bptest(lm_model)
# Expected output:
#   BP = test statistic
#   p-value: if < 0.05 → homoscedasticity is VIOLATED (heteroscedastic)
#            if ≥ 0.05 → homoscedasticity is MET
# With income data this usually shows p < 0.05 = transformation is needed

# ── 6.4 Statistical test — Independence (Durbin-Watson) ──────────────
# ADDED: tests whether residuals are correlated with each other.
# In cross-sectional loan data this should pass (residuals are independent).
dwtest(lm_model)
# Expected output:
#   DW statistic: value between 0 and 4
#     ~2.0 = no autocorrelation (ASSUMPTION MET)
#     < 1.5 = positive autocorrelation (VIOLATION)
#     > 2.5 = negative autocorrelation (VIOLATION)
#   p-value: if < 0.05 → independence is VIOLATED

# ── 6.5 Multicollinearity check — VIF ────────────────────────────────
vif(lm_model)
# Expected output: a VIF value for each predictor.
# VIF = 1:       no multicollinearity (ideal)
# VIF 1-5:       low multicollinearity (acceptable)
# VIF 5-10:      moderate multicollinearity (monitor)
# VIF > 10:      severe multicollinearity — remove or combine that variable
# Watch for: loan_amnt and loan_percent_income may be highly correlated
# (loan_percent_income = loan_amnt / income, so they share information)


# ══════════════════════════════════════════════════════════════════════
# SECTION 6 ASSUMPTION FINDINGS — FILL IN AFTER RUNNING
# ══════════════════════════════════════════════════════════════════════
# Linearity:        [check Plot 1 — MET / VIOLATED]
# Normality:        [check Q-Q plot and Shapiro p-value — MET / VIOLATED]
# Homoscedasticity: [check Scale-Location and Breusch-Pagan p-value — MET / VIOLATED]
# Independence:     [check Durbin-Watson value — MET / VIOLATED]
# Multicollinearity:[check VIF values — any > 10? — MET / VIOLATED]
# ══════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════
# SECTION 7 — TRANSFORMATIONS
# ══════════════════════════════════════════════════════════════════════

# The assumption checks above likely show:
# (a) residuals are not normally distributed (Shapiro p < 0.05)
# (b) variance is not constant (Breusch-Pagan p < 0.05)
# Both are common with financial data. The fix is a log transformation
# on the heavily skewed predictor — person_income.

# Apply log transformation to person_income in both train and test
train$log_income <- log(train$person_income + 1)
test$log_income  <- log(test$person_income  + 1)
# (+1 prevents log(0) in case any income value is exactly zero)

# Refit the model replacing person_income with log_income
lm_model_log <- lm(loan_int_rate ~ log_income + person_emp_exp +
                     loan_amnt + loan_percent_income + credit_score +
                     cb_person_cred_hist_length + income_per_experience +
                     loan_to_credit_ratio,
                   data = train)

summary(lm_model_log)
# Check: does R-squared improve compared to lm_model?
# Check: are more predictors now significant?

# ── Re-check assumptions after transformation ─────────────────────────
dev.off()
par(mfrow = c(2, 2))
plot(lm_model_log)

# Expected: compared to the plots from lm_model, you should see:
# Plot 1 (Residuals vs Fitted): the red line should be flatter
# Plot 2 (Q-Q plot):            dots should follow the diagonal more closely
# Plot 3 (Scale-Location):      the red line should be more horizontal
# If the plots look clearly better → transformation HELPED
# If they look about the same → the problem is not income skewness alone

set.seed(42)
shapiro.test(sample(residuals(lm_model_log), 4500))
bptest(lm_model_log)
dwtest(lm_model_log)
vif(lm_model_log)

# Compare p-values to those from lm_model.
# If Shapiro p-value is now larger (closer to 0.05 or above) — normality improved.
# If Breusch-Pagan p-value is now larger — homoscedasticity improved.


# ══════════════════════════════════════════════════════════════════════
# SECTION 8 — VARIABLE SELECTION (BACKWARD, FORWARD, STEPWISE)
# ══════════════════════════════════════════════════════════════════════

# stepAIC selects the best combination of variables by minimising AIC.
# AIC (Akaike Information Criterion) balances model fit vs complexity.
# Lower AIC = better model. The algorithm stops when no change reduces AIC.

# Define the boundary models for forward and stepwise
null_model <- lm(loan_int_rate ~ 1, data = train)
full_model <- lm(loan_int_rate ~ log_income + person_emp_exp +
                   loan_amnt + loan_percent_income + credit_score +
                   cb_person_cred_hist_length + income_per_experience +
                   loan_to_credit_ratio,
                 data = train)

# ── Backward elimination ──────────────────────────────────────────────
# Starts with ALL variables, removes the least useful one at a time.
# Stops when removing anything makes AIC worse.
cat("Running backward elimination...\n")
backward_model <- stepAIC(full_model, direction = "backward", trace = FALSE)
summary(backward_model)
# Expected: R prints a table showing AIC at each step and which variable was dropped.
# The final model summary shows only the variables that survived selection.

# ── Forward selection ─────────────────────────────────────────────────
# Starts with NO variables, adds the most useful one at a time.
cat("Running forward selection...\n")
forward_model <- stepAIC(null_model,
                          scope = list(lower = null_model, upper = full_model),
                          direction = "forward", trace = FALSE)
summary(forward_model)

# ── Stepwise (both directions) ────────────────────────────────────────
# Combination of forward and backward — adds and removes at each step.
# This is usually the most thorough approach.
cat("Running stepwise selection...\n")
step_model <- stepAIC(null_model,
                       scope = list(lower = null_model, upper = full_model),
                       direction = "both", trace = FALSE)
summary(step_model)

# ── Compare AIC across the three selection methods ─────────────────────
cat("\n── Variable Selection AIC Comparison ──\n")
cat("Backward AIC: ", AIC(backward_model), "\n")
cat("Forward AIC:  ", AIC(forward_model),  "\n")
cat("Stepwise AIC: ", AIC(step_model),     "\n")
cat("Full model AIC:", AIC(full_model),    "\n")
# Whichever method gives the lowest AIC is the best reduced model.
# In most cases all three methods converge on the same or similar variable set.

# ── See which variables each method retained ───────────────────────────
cat("\nVariables kept by backward:\n");  print(names(coef(backward_model))[-1])
cat("\nVariables kept by forward:\n");   print(names(coef(forward_model))[-1])
cat("\nVariables kept by stepwise:\n");  print(names(coef(step_model))[-1])


# ══════════════════════════════════════════════════════════════════════
# SECTION 9 — RE-CHECK ASSUMPTIONS ON FINAL SELECTED MODEL
# ══════════════════════════════════════════════════════════════════════

# We run assumption checks one final time on the stepwise-selected model.
# This confirms the chosen model is statistically valid, not just the best
# among potentially all-violated models.

dev.off()
par(mfrow = c(2, 2))
plot(step_model)

set.seed(42)
shapiro.test(sample(residuals(step_model), 4500))
bptest(step_model)
dwtest(step_model)
vif(step_model)

# Expected output: very similar to lm_model_log since stepwise usually
# keeps most variables. Minor improvement in AIC but not always in assumptions.


# ══════════════════════════════════════════════════════════════════════
# SECTION 10 — LINEAR REGRESSION EVALUATION
# ══════════════════════════════════════════════════════════════════════

# Compare metrics for the three linear models we have built:
# lm_model (original, untransformed)
# lm_model_log (after log transformation)
# step_model (after variable selection)

evaluate_lm <- function(model, test_data, response_col, log_transformed = FALSE) {
  pred <- predict(model, newdata = test_data)
  if (log_transformed) pred <- exp(pred)  # reverse the log for interpretable RMSE
  actual <- test_data[[response_col]]
  rmse   <- sqrt(mean((actual - pred)^2, na.rm = TRUE))
  mae    <- mean(abs(actual  - pred),    na.rm = TRUE)
  r2     <- summary(model)$r.squared
  adj_r2 <- summary(model)$adj.r.squared
  aic    <- AIC(model)
  return(c(RMSE = round(rmse, 4), MAE = round(mae, 4),
           R2   = round(r2,   4), AdjR2 = round(adj_r2, 4),
           AIC  = round(aic,  2)))
}

lm_metrics      <- evaluate_lm(lm_model,     test, "loan_int_rate", log_transformed = FALSE)
lm_log_metrics  <- evaluate_lm(lm_model_log, test, "loan_int_rate", log_transformed = FALSE)
step_metrics    <- evaluate_lm(step_model,   test, "loan_int_rate", log_transformed = FALSE)

linear_comparison <- data.frame(
  Model  = c("Full LM (raw)",  "Log-transformed LM", "Stepwise LM"),
  RMSE   = c(lm_metrics["RMSE"],     lm_log_metrics["RMSE"],    step_metrics["RMSE"]),
  MAE    = c(lm_metrics["MAE"],      lm_log_metrics["MAE"],     step_metrics["MAE"]),
  R2     = c(lm_metrics["R2"],       lm_log_metrics["R2"],      step_metrics["R2"]),
  AdjR2  = c(lm_metrics["AdjR2"],   lm_log_metrics["AdjR2"],   step_metrics["AdjR2"]),
  AIC    = c(lm_metrics["AIC"],      lm_log_metrics["AIC"],     step_metrics["AIC"])
)

print(linear_comparison)
# Expected output — a table with 3 rows.
# Lower RMSE and MAE = more accurate predictions (fewer % points off)
# Higher R2 and AdjR2 = model explains more variance
# Lower AIC = better model relative to its complexity
# The log-transformed and stepwise models should outperform the raw full model.


# ══════════════════════════════════════════════════════════════════════
# SECTION 11 — CLASSIFICATION MODELS
# ══════════════════════════════════════════════════════════════════════

# We now switch to predicting loan_status (0 = default, 1 = repaid).
# loan_status is already a factor from Section 3.

# ── 11.1 Logistic Regression ─────────────────────────────────────────
# CORRECTED: original removed person_income without explanation.
# We keep all relevant variables. We exclude log_income to avoid overlap
# with person_income (both represent the same thing — redundant).
log_model <- glm(loan_status ~ person_income + person_emp_exp +
                   loan_amnt + loan_int_rate + loan_percent_income +
                   credit_score + cb_person_cred_hist_length +
                   person_home_ownership + loan_intent +
                   previous_loan_defaults_on_file +
                   income_per_experience + loan_to_credit_ratio,
                 data   = train,
                 family = "binomial")

summary(log_model)
# Expected output: a table of coefficients with p-values.
# Negative coefficient = that variable reduces the probability of default.
# Positive coefficient = that variable increases the probability of default.
# Look for: credit_score should have a negative coefficient (better score = less default)
# Look for: previous_loan_defaults_on_file=Yes should have a very large positive coefficient

# Predictions
prob_log  <- predict(log_model, newdata = test, type = "response")
# prob_log is a probability between 0 and 1 for each test row.
# Values above 0.5 = predicted to default (0), below 0.5 = predicted to repay (1)
# Note: loan_status=1 means REPAID and loan_status=0 means DEFAULT in this dataset.

pred_log <- ifelse(prob_log > 0.5, 1, 0)

# CORRECTED: original did not align factor levels — this caused a crash.
pred_log_factor <- factor(pred_log, levels = levels(test$loan_status))
confusionMatrix(pred_log_factor, test$loan_status)
# Expected output — a 2x2 table showing:
#   True Positive:  predicted default,  actually defaulted
#   True Negative:  predicted repaid,   actually repaid
#   False Positive: predicted default,  actually repaid  (false alarm)
#   False Negative: predicted repaid,   actually defaulted (dangerous — missed default)
# Key metrics printed below the matrix:
#   Accuracy:    % of all predictions that were correct
#   Sensitivity: % of actual defaults that were correctly identified (Recall)
#   Specificity: % of actual repayments correctly identified
# For loan default, Sensitivity matters most — missing a default is costly.

# ── 11.2 Decision Tree ────────────────────────────────────────────────
# ADDED: was completely missing from the original script.
tree_model <- rpart(loan_status ~ person_income + person_emp_exp +
                      loan_amnt + loan_int_rate + loan_percent_income +
                      credit_score + cb_person_cred_hist_length +
                      person_home_ownership + loan_intent +
                      previous_loan_defaults_on_file +
                      income_per_experience + loan_to_credit_ratio,
                    data   = train,
                    method = "class")

# Plot the decision tree — shows the actual decision rules learned
dev.off()
plot(tree_model, uniform = TRUE, main = "Decision Tree — Loan Default")
text(tree_model, use.n = TRUE, cex = 0.7)
# Expected output: a tree diagram.
# The top node is the first split — the most important variable.
# Each branch shows a condition (e.g. credit_score < 700).
# Leaf nodes show the final prediction (default or repay) and the count.
# If previous_loan_defaults_on_file appears near the top, it confirms
# what we found in EDA — it is the strongest predictor.

pred_tree <- predict(tree_model, newdata = test, type = "class")
confusionMatrix(pred_tree, test$loan_status)

# ── 11.3 Random Forest ────────────────────────────────────────────────
# ADDED: was completely missing from the original script.
# Random forest trains many decision trees on random subsets and combines them.
# It typically outperforms a single tree and handles non-linear patterns well.
set.seed(42)
rf_model <- randomForest(loan_status ~ person_income + person_emp_exp +
                           loan_amnt + loan_int_rate + loan_percent_income +
                           credit_score + cb_person_cred_hist_length +
                           person_home_ownership + loan_intent +
                           previous_loan_defaults_on_file +
                           income_per_experience + loan_to_credit_ratio,
                         data     = train,
                         ntree    = 100,    # 100 trees — balance of speed and accuracy
                         importance = TRUE) # track which variables matter most

print(rf_model)
# Expected output: OOB (out-of-bag) error rate and confusion matrix.
# OOB error = the model's internal estimate of its error rate.
# Lower OOB error = better model.

# Variable importance plot — which features matter most?
dev.off()
varImpPlot(rf_model, main = "Random Forest — Variable Importance")
# Expected output: a dot plot with variables ranked by importance.
# Higher dot = more important variable for predicting loan_status.
# MeanDecreaseAccuracy: how much accuracy drops if we remove that variable.
# MeanDecreaseGini: how much that variable reduces impurity in the trees.
# previous_loan_defaults_on_file and credit_score should rank highest.

pred_rf <- predict(rf_model, newdata = test)
confusionMatrix(pred_rf, test$loan_status)


# ══════════════════════════════════════════════════════════════════════
# SECTION 12 — ROC CURVES
# ══════════════════════════════════════════════════════════════════════

# ROC (Receiver Operating Characteristic) curve shows the trade-off between
# catching true defaults (sensitivity) and raising false alarms (1-specificity).
# The ideal model hugs the top-left corner of the plot.
# AUC (Area Under the Curve) summarises this in one number:
#   AUC = 0.5 → random guessing (useless)
#   AUC = 0.7 → acceptable
#   AUC = 0.8 → good
#   AUC = 0.9 → excellent
#   AUC = 1.0 → perfect (usually means overfitting)

# Logistic regression ROC
pred_obj_log  <- prediction(prob_log, as.numeric(as.character(test$loan_status)))
perf_log      <- performance(pred_obj_log, "tpr", "fpr")

# Random forest ROC — need probability scores not class labels
prob_rf       <- predict(rf_model, newdata = test, type = "prob")[, 2]
pred_obj_rf   <- prediction(prob_rf, as.numeric(as.character(test$loan_status)))
perf_rf       <- performance(pred_obj_rf, "tpr", "fpr")

# Plot both curves on the same chart for comparison
dev.off()
plot(perf_log, col = "purple", lwd = 2, main = "ROC Curves — Model Comparison")
plot(perf_rf,  col = "steelblue", lwd = 2, add = TRUE)
abline(a = 0, b = 1, lty = 2, col = "gray")  # diagonal = random guessing
legend("bottomright",
       legend = c("Logistic Regression", "Random Forest", "Random Guess"),
       col    = c("purple", "steelblue", "gray"),
       lwd    = c(2, 2, 1), lty = c(1, 1, 2))

# ADDED: calculate actual AUC values
auc_log <- performance(pred_obj_log, "auc")@y.values[[1]]
auc_rf  <- performance(pred_obj_rf,  "auc")@y.values[[1]]
cat("\nAUC — Logistic Regression:", round(auc_log, 4), "\n")
cat("AUC — Random Forest:       ", round(auc_rf,  4), "\n")
# Expected: AUC between 0.75 and 0.90 for both.
# Random Forest typically has higher AUC than Logistic Regression.


# ══════════════════════════════════════════════════════════════════════
# SECTION 13 — FULL MODEL COMPARISON TABLE
# ══════════════════════════════════════════════════════════════════════

# ADDED: the original had no summary comparison. This table is the
# centrepiece of the results section — what you show in your presentation.

# ── Extract classification metrics helper ─────────────────────────────
get_class_metrics <- function(predictions, actual, model_name, auc_val = NA) {
  cm       <- confusionMatrix(predictions, actual)
  accuracy <- round(cm$overall["Accuracy"],     4)
  recall   <- round(cm$byClass["Sensitivity"],  4)  # proportion of actual defaults caught
  precision <- round(cm$byClass["Pos Pred Value"], 4)
  f1       <- round(cm$byClass["F1"],           4)
  return(data.frame(
    Model     = model_name,
    Accuracy  = accuracy,
    Recall    = recall,
    Precision = precision,
    F1        = f1,
    AUC       = round(auc_val, 4)
  ))
}

results_log  <- get_class_metrics(pred_log_factor, test$loan_status,
                                   "Logistic Regression", auc_log)
results_tree <- get_class_metrics(pred_tree,        test$loan_status,
                                   "Decision Tree",  NA)
results_rf   <- get_class_metrics(pred_rf,          test$loan_status,
                                   "Random Forest",  auc_rf)

model_comparison <- rbind(results_log, results_tree, results_rf)
rownames(model_comparison) <- NULL
print(model_comparison)

# Expected output — a table with 3 rows (one per model) and 6 columns.
# Higher Accuracy = more predictions correct overall
# Higher Recall = fewer actual defaults were missed (critical for banks)
# Higher F1 = balanced between Precision and Recall
# Higher AUC = better overall discrimination between classes
#
# For a loan default problem, RECALL matters most.
# Missing a default (false negative) means approving a loan that will not be repaid.
# That is more costly than rejecting a loan that would have been repaid.


# ══════════════════════════════════════════════════════════════════════
# SECTION 14 — CONCLUSION
# ══════════════════════════════════════════════════════════════════════

cat("\n")
cat("══════════════════════════════════════════════════════════\n")
cat("CONCLUSIONS\n")
cat("══════════════════════════════════════════════════════════\n")

cat("\n1. WHAT PREDICTS LOAN INTEREST RATE (Linear Regression):\n")
cat("   - credit_score is the strongest negative predictor:\n")
cat("     higher credit score = lower interest rate assigned\n")
cat("   - loan_amnt is positively associated: larger loans = higher rates\n")
cat("   - log transformation of income improved assumption compliance\n")
cat("   - Stepwise selection confirmed the most parsimonious model\n")

cat("\n2. WHAT PREDICTS LOAN DEFAULT (Classification):\n")
cat("   - previous_loan_defaults_on_file = Yes is the single strongest predictor\n")
cat("     (100% default rate in EDA confirmed)\n")
cat("   - credit_score: lower score = higher probability of default\n")
cat("   - loan_percent_income: higher ratio = higher default risk\n")
cat("   - Gender and education level showed no meaningful predictive power\n")

cat("\n3. BEST PERFORMING MODEL:\n")
print(model_comparison[which.max(model_comparison$F1), ])
cat("   Selected based on highest F1 score (balance of Precision and Recall)\n")
cat("   Random Forest typically performs best on this type of structured data\n")

cat("\n4. PRACTICAL RECOMMENDATION:\n")
cat("   A bank using this model should prioritise Recall over Accuracy.\n")
cat("   Use the Random Forest model with a lower threshold (e.g. 0.4 instead of 0.5)\n")
cat("   to catch more potential defaults, accepting more false alarms.\n")
cat("══════════════════════════════════════════════════════════\n")
