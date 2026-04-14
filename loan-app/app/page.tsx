"use client";
import { useState } from "react";

const sections = [
  { id: "overview", label: "Overview", icon: "◈" },
  { id: "eda", label: "EDA", icon: "◉" },
  { id: "linear", label: "Linear Regression", icon: "◧" },
  { id: "logistic", label: "Logistic Regression", icon: "◨" },
  { id: "features", label: "Feature Engineering", icon: "◫" },
];

const codeBlocks: Record<string, { title: string; code: string; output?: string }[]> = {
  overview: [
    {
      title: "Install & Load Libraries",
      code: `install.packages(c("tidyverse", "caret", "corrplot", "car", "MASS", "ROCR"))

library(tidyverse)
library(caret)
library(corrplot)
library(car)
library(MASS)
library(ROCR)
library(dplyr)`,
    },
    {
      title: "Load Dataset",
      code: `data <- read.csv("/content/sample_data/loan_data.csv")

head(data)
summary(data)`,
      output: `Rows: 45,000  |  Columns: 14\nVariables: person_age, person_gender, person_education,\n           person_income, person_emp_exp, person_home_ownership,\n           loan_amnt, loan_intent, loan_int_rate,\n           loan_percent_income, cb_person_cred_hist_length,\n           credit_score, previous_loan_defaults_on_file, loan_status`,
    },
    {
      title: "Data Cleaning",
      code: `colSums(is.na(data))

data$person_gender              <- as.factor(data$person_gender)
data$person_education           <- as.factor(data$person_education)
data$person_home_ownership      <- as.factor(data$person_home_ownership)
data$loan_intent                <- as.factor(data$loan_intent)
data$previous_loan_defaults_on_file <- as.factor(data$previous_loan_defaults_on_file)
data$loan_status                <- as.factor(data$loan_status)

summary(data)`,
      output: `No missing values detected across all 14 columns.\n6 categorical columns converted to factors.`,
    },
  ],
  eda: [
    {
      title: "Correlation Matrix",
      code: `numeric_data <- data %>% dplyr::select(where(is.numeric))

cor_matrix <- cor(numeric_data)
corrplot(cor_matrix, method = "color")`,
      output: `Notable correlations:\n  loan_amnt     <-> loan_percent_income  : 0.64\n  credit_score  <-> loan_int_rate        : -0.37\n  person_age    <-> person_emp_exp       : 0.82  <- multicollinearity risk`,
    },
    {
      title: "Income Distribution",
      code: `hist(data$person_income, breaks = 30, main = "Income Distribution")
# Right-skewed: most people earn low-to-moderate income`,
      output: `Distribution: Right-skewed\n-> Log transformation will be applied to normalise for modelling`,
    },
  ],
  linear: [
    {
      title: "Train/Test Split (80/20)",
      code: `set.seed(123)  # reproducibility

train_index <- createDataPartition(data$person_income, p = 0.8, list = FALSE)

train <- data[train_index, ]
test  <- data[-train_index, ]`,
      output: `Training set: 36,000 rows (80%)\nTest set    :  9,000 rows (20%)`,
    },
    {
      title: "Baseline Linear Model",
      code: `lm_model <- lm(person_income ~ ., data = train)

summary(lm_model)`,
      output: `R² = 0.2927  (explains ~29% of variance)`,
    },
    {
      title: "Check Assumptions & Multicollinearity",
      code: `par(mfrow = c(2,2))
plot(lm_model)

vif(lm_model)`,
      output: `VIF Results:\n  person_age      GVIF = 14.14  <- HIGH\n  person_emp_exp  GVIF = 11.36  <- HIGH\n  Others          GVIF < 5      OK`,
    },
    {
      title: "Log Transform Income",
      code: `train$log_income <- log(train$person_income)
test$log_income  <- log(test$person_income)

lm_model_log <- lm(log_income ~ . - person_income, data = train)
summary(lm_model_log)`,
      output: `R² = 0.7666  (major improvement — explains ~77% of variance)`,
    },
    {
      title: "Evaluate on Test Set",
      code: `pred     <- predict(lm_model_log, newdata = test)
pred_exp <- exp(pred)  # back-transform

RMSE <- sqrt(mean((test$person_income - pred_exp)^2))
MAE  <- mean(abs(test$person_income - pred_exp))

RMSE
MAE`,
      output: `RMSE = 49,141.56\nMAE  = 16,591.53`,
    },
    {
      title: "Stepwise Regression (AIC)",
      code: `step_model <- stepAIC(lm_model_log, direction = "both")
summary(step_model)`,
      output: `Removed: credit_score, person_gender, previous_loan_defaults_on_file\nFinal R² = 0.7666  (parsimonious model)`,
    },
  ],
  logistic: [
    {
      title: "Logistic Regression Model",
      code: `log_model <- glm(loan_status ~ . - person_income,
                  data   = train,
                  family = "binomial")

summary(log_model)`,
      output: `Target: loan_status (0 = no default, 1 = default)\nAIC: 18,432`,
    },
    {
      title: "Predictions & Confusion Matrix",
      code: `prob       <- predict(log_model, newdata = test, type = "response")
pred_class <- ifelse(prob > 0.5, 1, 0)

confusionMatrix(as.factor(pred_class), test$loan_status)`,
      output: `Accuracy    = 89.79%\nSensitivity = 93.77%\nSpecificity = 76.02%`,
    },
    {
      title: "ROC Curve",
      code: `pred_obj <- prediction(prob, test$loan_status)
perf     <- performance(pred_obj, "tpr", "fpr")

plot(perf, col = "purple", main = "ROC Curve")`,
      output: `AUC ~= 0.93  — Excellent discriminative ability`,
    },
  ],
  features: [
    {
      title: "Engineer New Features",
      code: `# Income relative to work experience
data$income_per_experience <- data$person_income / (data$person_emp_exp + 1)

# Loan amount relative to credit score
data$loan_to_credit_ratio  <- data$loan_amnt / data$credit_score`,
      output: `2 new features created:\n  income_per_experience  — earning efficiency\n  loan_to_credit_ratio   — credit risk relative to loan`,
    },
  ],
};

const metrics = [
  { label: "Loan Records", value: "45K" },
  { label: "Features", value: "14" },
  { label: "Linear R²", value: "76.7%" },
  { label: "Accuracy", value: "89.8%" },
];

export default function Home() {
  const [active, setActive] = useState("overview");
  const [copied, setCopied] = useState<string | null>(null);

  const handleCopy = (code: string, id: string) => {
    navigator.clipboard.writeText(code);
    setCopied(id);
    setTimeout(() => setCopied(null), 1500);
  };

  return (
    <div className="min-h-screen bg-[#08080f] text-[#e0dbd0] font-mono">
      <header className="border-b border-[#1a1a28] px-8 py-5 flex items-center justify-between sticky top-0 bg-[#08080f]/95 backdrop-blur z-40">
        <div>
          <div className="text-[9px] tracking-[0.5em] text-[#5a5a7a] uppercase mb-1">R · Statistical Modelling</div>
          <h1 className="text-lg font-bold tracking-tight">
            LOAN DATA<span className="text-[#7c6fff]"> ANALYSIS</span>
          </h1>
        </div>
        <div className="flex gap-8">
          {metrics.map((m) => (
            <div key={m.label} className="text-right">
              <div className="text-xl font-bold text-[#7c6fff] tabular-nums">{m.value}</div>
              <div className="text-[8px] tracking-[0.3em] text-[#3a3a5a] uppercase">{m.label}</div>
            </div>
          ))}
        </div>
      </header>

      <div className="flex">
        <nav className="w-48 border-r border-[#1a1a28] min-h-[calc(100vh-69px)] p-3 shrink-0 sticky top-[69px] self-start">
          <div className="text-[8px] tracking-[0.4em] text-[#3a3a5a] uppercase mb-3 px-2">Navigation</div>
          {sections.map((s) => (
            <button
              key={s.id}
              onClick={() => setActive(s.id)}
              className={`w-full text-left px-3 py-2 rounded text-[12px] flex items-center gap-2 transition-all mb-0.5 ${
                active === s.id
                  ? "bg-[#7c6fff]/10 text-[#7c6fff] border border-[#7c6fff]/20"
                  : "text-[#5a5a7a] hover:text-[#e0dbd0] hover:bg-[#12121e]"
              }`}
            >
              <span className="text-[10px]">{s.icon}</span>
              {s.label}
            </button>
          ))}
          <div className="mt-6 px-2 space-y-1">
            <div className="text-[8px] tracking-[0.4em] text-[#3a3a5a] uppercase mb-2">Packages</div>
            {["tidyverse", "caret", "corrplot", "car", "MASS", "ROCR"].map((lib) => (
              <div key={lib} className="text-[10px] text-[#3a3a5a] py-0.5 border-b border-[#1a1a28]">{lib}</div>
            ))}
          </div>
        </nav>

        <main className="flex-1 p-8 max-w-3xl">
          <div className="text-[9px] tracking-[0.4em] text-[#3a3a5a] uppercase mb-5">
            {sections.find((s) => s.id === active)?.label}
          </div>

          <div className="space-y-5">
            {(codeBlocks[active] || []).map((block, i) => (
              <div key={i} className="border border-[#1a1a28] rounded-lg overflow-hidden group">
                <div className="flex items-center justify-between bg-[#0c0c18] px-4 py-2.5 border-b border-[#1a1a28]">
                  <span className="text-[11px] text-[#7c6fff]">{block.title}</span>
                  <button
                    onClick={() => handleCopy(block.code, `${active}-${i}`)}
                    className="text-[9px] tracking-widest text-[#3a3a5a] hover:text-[#e0dbd0] transition-colors"
                  >
                    {copied === `${active}-${i}` ? "✓ COPIED" : "COPY"}
                  </button>
                </div>
                <div className="bg-[#0a0a16] p-5">
                  <pre className="text-[12px] leading-relaxed text-[#c0bba8] whitespace-pre-wrap"><code>{block.code}</code></pre>
                </div>
                {block.output && (
                  <div className="bg-[#060610] border-t border-[#1a1a28] px-5 py-3">
                    <div className="text-[8px] tracking-[0.4em] text-[#3a3a5a] uppercase mb-2">Output</div>
                    <pre className="text-[11px] text-[#4a8e5f] leading-relaxed whitespace-pre-wrap">{block.output}</pre>
                  </div>
                )}
              </div>
            ))}
          </div>

          {active === "linear" && (
            <div className="mt-6 grid grid-cols-3 gap-3">
              {[
                { label: "Baseline R²", val: "29.3%", note: "before transform" },
                { label: "Post-log R²", val: "76.7%", note: "+47pp improvement" },
                { label: "Test MAE", val: "16,591", note: "on original scale" },
              ].map((c) => (
                <div key={c.label} className="border border-[#1a1a28] rounded-lg p-4 bg-[#0c0c18]">
                  <div className="text-2xl font-bold text-[#7c6fff]">{c.val}</div>
                  <div className="text-[10px] text-[#e0dbd0] mt-1">{c.label}</div>
                  <div className="text-[9px] text-[#3a3a5a] mt-0.5">{c.note}</div>
                </div>
              ))}
            </div>
          )}

          {active === "logistic" && (
            <div className="mt-6 grid grid-cols-3 gap-3">
              {[
                { label: "Accuracy", val: "89.8%", col: "#4a8e5f" },
                { label: "Sensitivity", val: "93.8%", col: "#4a8e5f" },
                { label: "AUC", val: "≈0.93", col: "#4a8e5f" },
              ].map((c) => (
                <div key={c.label} className="border border-[#1a1a28] rounded-lg p-4 bg-[#0c0c18]">
                  <div className="text-2xl font-bold" style={{ color: c.col }}>{c.val}</div>
                  <div className="text-[10px] text-[#e0dbd0] mt-1">{c.label}</div>
                </div>
              ))}
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
