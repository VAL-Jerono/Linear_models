"use client";
import { useState } from "react";

const sections = [
  { id: "problem", label: "Business Problem", icon: "◈" },
  { id: "data", label: "Data Understanding", icon: "◉" },
  { id: "preparation", label: "Data Preparation", icon: "◫" },
  { id: "income", label: "Income Verification", icon: "◧" },
  { id: "credit", label: "Credit Risk", icon: "◨" },
  { id: "conclusion", label: "Decision Support", icon: "◎" },
];

const tagColors: Record<string, string> = {
  REGRESSION: "#7c6fff",
  CLASSIFICATION: "#4a8e5f",
  STRATEGY: "#5a7ab5",
  FINDING: "#d4a843",
  RISK: "#e05252",
  INSIGHT: "#7c6fff",
  "KEY SIGNAL": "#e05252",
  METHODOLOGY: "#5a7ab5",
  SPLIT: "#4a8e5f",
  "MODEL SELECTION": "#7c6fff",
  LIMITATION: "#d4a843",
  APPLICATION: "#4a8e5f",
  "MODEL QUALITY": "#7c6fff",
  "THRESHOLD DECISION": "#d4a843",
  EXPLAINABILITY: "#5a7ab5",
  DEPLOYMENT: "#4a8e5f",
};

const headerMetrics = [
  { label: "Borrower Records", value: "45K" },
  { label: "Income R²", value: "76.7%" },
  { label: "Classifier AUC", value: "0.93" },
  { label: "Test Accuracy", value: "89.8%" },
];

type Finding = { label: string; value: string; note?: string; highlight?: boolean };
type InsightBlock = { title: string; body: string; tag?: string };
type MetricCard = { label: string; val: string; sub: string; color?: string };
type RiskFactor = { factor: string; level: string; color: string };
type ConfusionMatrix = { tn: number; fp: number; fn: number; tp: number };
type CodeBlock = { title: string; code: string; output: string };

type SectionContent = {
  headline: string;
  subline: string;
  findings?: Finding[];
  insights?: InsightBlock[];
  codeBlock?: CodeBlock;
  metrics?: MetricCard[];
  riskFactors?: RiskFactor[];
  matrix?: ConfusionMatrix;
};

const content: Record<string, SectionContent> = {
  problem: {
    headline: "Two questions before every loan approval.",
    subline:
      "Our credit team evaluates thousands of applications monthly. Before approving any facility, we need answers to two core questions: Is the declared income plausible? And how likely is this borrower to repay? This project builds predictive models that answer both — at scale.",
    insights: [
      {
        title: "Task 1 — Income Verification",
        body: "Applicants self-report income. Our regression model predicts expected income from borrower profile variables. A large gap between declared and predicted income flags an application for manual review before a decision is issued.",
        tag: "REGRESSION",
      },
      {
        title: "Task 2 — Repayment Risk Scoring",
        body: "Our classification model scores each applicant on repayment likelihood. High-risk profiles are routed for enhanced due diligence or declined. Low-risk profiles move to fast-track approval without delays.",
        tag: "CLASSIFICATION",
      },
      {
        title: "Why Both Models Work Together",
        body: "A borrower may have believable income but poor repayment behavior, or vice versa. Using both models gives the credit desk a complete picture — income plausibility and default probability — before committing capital.",
        tag: "STRATEGY",
      },
    ],
  },
  data: {
    headline: "45,000 borrower records. 14 variables per applicant.",
    subline:
      "Our dataset covers applicant demographics, employment history, loan details, credit history, and past default behavior. Before modeling, we characterized the data to identify patterns, risks, and structural issues that would affect model performance.",
    findings: [
      { label: "Total Records", value: "45,000", note: "borrower applications" },
      { label: "Predictor Variables", value: "14", note: "per applicant" },
      { label: "Default Rate", value: "22.4%", note: "class imbalance noted" },
      { label: "Missing Values", value: "None", note: "across all 14 columns", highlight: true },
    ],
    insights: [
      {
        title: "Income is heavily right-skewed",
        body: "Most applicants earn low-to-moderate income with a long tail of high earners. A direct regression on raw income violates OLS normality assumptions — log transformation is required before fitting any linear model.",
        tag: "FINDING",
      },
      {
        title: "Age and experience are collinear — VIF 14.14 and 11.36",
        body: "Person age (GVIF = 14.14) and employment experience (GVIF = 11.36) are highly correlated with each other. This multicollinearity inflates standard errors in linear models. Both were retained for predictive value, but coefficient interpretation requires care.",
        tag: "RISK",
      },
      {
        title: "Credit score inversely tracks interest rate",
        body: "Correlation of −0.37 between credit score and loan interest rate confirms that lenders are already pricing in creditworthiness at origination. This variable carries strong signal in the classification model.",
        tag: "INSIGHT",
      },
      {
        title: "Previous defaults are the single strongest risk signal",
        body: "Borrowers with a prior default on file show dramatically different repayment outcomes from the rest of the population. This binary variable carries disproportionate predictive weight and appears in the top features of all classification models tested.",
        tag: "KEY SIGNAL",
      },
    ],
  },
  preparation: {
    headline: "Clean data in. Reliable predictions out.",
    subline:
      "Raw borrower data required preparation before modeling. We removed biologically impossible outliers, engineered affordability features that the raw variables do not capture directly, and encoded categoricals for model ingestion.",
    codeBlock: {
      title: "Feature Engineering — Affordability and Credit Ratios",
      code: `# Earning efficiency: income relative to work experience
data$income_per_experience <- data$person_income / (data$person_emp_exp + 1)

# Credit utilisation proxy
data$loan_to_credit_ratio  <- data$loan_amnt / data$credit_score

# Encode categoricals as factors
data$person_gender              <- as.factor(data$person_gender)
data$person_education           <- as.factor(data$person_education)
data$person_home_ownership      <- as.factor(data$person_home_ownership)
data$loan_intent                <- as.factor(data$loan_intent)
data$previous_loan_defaults_on_file <- as.factor(data$previous_loan_defaults_on_file)
data$loan_status                <- as.factor(data$loan_status)

# Train / Test split — 80/20, stratified
set.seed(123)
train_index <- createDataPartition(data$person_income, p = 0.8, list = FALSE)
train <- data[train_index, ]
test  <- data[-train_index, ]`,
      output: `Engineered: income_per_experience, loan_to_credit_ratio
All categorical columns converted to factors. No new missing values introduced.
Age > 100 removed (biologically impossible). High-income and high-experience records retained.

Training set: 36,000 records  |  Test set: 9,000 records`,
    },
    insights: [
      {
        title: "Outlier strategy: surgical, not aggressive",
        body: "Only age values above 100 were removed as biologically impossible. High-income and long-tenure records were deliberately retained — extreme values are real in a lending population and carry predictive information about the upper tail.",
        tag: "METHODOLOGY",
      },
      {
        title: "Seed-controlled split ensures reproducibility",
        body: "set.seed(123) guarantees the same train/test partition on every run. Any analyst reproducing this work will get identical results, which is essential for auditing credit model decisions.",
        tag: "SPLIT",
      },
    ],
  },
  income: {
    headline: "Can we trust what the applicant declared?",
    subline:
      "The regression model predicts expected income from borrower profile variables. A large residual — declared income far above the model's prediction — is a plausibility flag. We compared multiple models; post-transformation linear regression delivered the best interpretability-performance balance.",
    metrics: [
      { label: "Baseline R²", val: "29.3%", sub: "raw income, no transform", color: "#e05252" },
      { label: "Post-Transform R²", val: "76.7%", sub: "after log transformation", color: "#4a8e5f" },
      { label: "Test RMSE", val: "49,142", sub: "back-transformed to original scale", color: "#7c6fff" },
      { label: "Test MAE", val: "16,592", sub: "mean absolute error", color: "#7c6fff" },
    ],
    codeBlock: {
      title: "Log Transformation — Before and After",
      code: `# Raw income: strongly right-skewed — violates OLS normality assumption
# Baseline model R² = 0.2927 (only 29% of variance explained)

train$log_income <- log(train$person_income)
test$log_income  <- log(test$person_income)

# Refit on log scale
lm_model_log <- lm(log_income ~ . - person_income, data = train)
summary(lm_model_log)
# R² jumps to 0.7666

# Back-transform predictions for business use
pred_exp <- exp(predict(lm_model_log, newdata = test))

RMSE <- sqrt(mean((test$person_income - pred_exp)^2))  # 49,141.56
MAE  <- mean(abs(test$person_income - pred_exp))       # 16,591.53

# Stepwise selection via AIC — removes credit_score, gender, prior defaults
step_model <- stepAIC(lm_model_log, direction = "both")
# Final parsimonious R² = 0.7666 (retained)`,
      output: `Before transform:  R² = 0.2927  (29% variance explained)
After log transform: R² = 0.7666  (77% variance explained) — +47 percentage points

RMSE = 49,141.56  |  MAE = 16,591.53  (original income scale)

Stepwise AIC removed: credit_score, person_gender, previous_loan_defaults_on_file
These variables add noise to income prediction — retained only in the classification model.`,
    },
    insights: [
      {
        title: "Transformation was not optional — it was necessary",
        body: "Without the log transformation, the OLS normality assumption is violated and the model explains only 29% of income variance. After transformation, the residuals become symmetric and the model captures 77% of variance. This is the single most impactful step in the regression pipeline.",
        tag: "MODEL SELECTION",
      },
      {
        title: "Multicollinearity is documented, not resolved",
        body: "Age (GVIF 14.14) and employment experience (GVIF 11.36) remain collinear in the final model. We chose to retain both for predictive power. Individual coefficient magnitudes should not be interpreted in isolation — use the model for prediction, not causal inference.",
        tag: "LIMITATION",
      },
      {
        title: "Business deployment: income anomaly flagging",
        body: "For any new application: if declared income exceeds the model's predicted income by more than 1.5 standard deviations on the log scale, the application is routed to a credit analyst for income document verification before a decision is issued.",
        tag: "APPLICATION",
      },
    ],
  },
  credit: {
    headline: "Repay or default? The model calls it first.",
    subline:
      "Our classification model scores each applicant on repayment likelihood. We evaluated Logistic Regression against tree-based alternatives. Logistic Regression is our primary model — it achieved AUC 0.93 and delivers full interpretability for regulatory reporting.",
    metrics: [
      { label: "Accuracy", val: "89.8%", sub: "9,000 held-out test records", color: "#4a8e5f" },
      { label: "Sensitivity", val: "93.8%", sub: "true repayer detection rate", color: "#4a8e5f" },
      { label: "Specificity", val: "76.0%", sub: "true default detection rate", color: "#d4a843" },
      { label: "AUC-ROC", val: "≈ 0.93", sub: "excellent risk separation", color: "#7c6fff" },
    ],
    matrix: { tn: 6546, fp: 435, fn: 484, tp: 1534 },
    riskFactors: [
      { factor: "Previous default on file", level: "CRITICAL", color: "#e05252" },
      { factor: "Debt-to-income ratio", level: "HIGH", color: "#d4a843" },
      { factor: "Loan interest rate tier", level: "HIGH", color: "#d4a843" },
      { factor: "Home ownership — RENT", level: "MODERATE", color: "#7c6fff" },
      { factor: "Credit score below 600", level: "MODERATE", color: "#7c6fff" },
      { factor: "Loan purpose — EDUCATION", level: "LOWER RISK", color: "#4a8e5f" },
    ],
    insights: [
      {
        title: "AUC 0.93 — strong separation across the full risk spectrum",
        body: "A model with AUC near 1.0 perfectly separates repayers from defaulters. Our 0.93 AUC means the model correctly ranks a randomly chosen repayer above a randomly chosen defaulter 93% of the time — far above the 0.50 baseline of a random classifier.",
        tag: "MODEL QUALITY",
      },
      {
        title: "High sensitivity protects against over-rejection",
        body: "The model correctly classifies 93.8% of borrowers who would repay — minimising the business cost of turning away creditworthy customers. The 76.0% specificity means some defaults are misclassified as safe; the 0.5 threshold can be raised to trade recall for precision depending on risk appetite.",
        tag: "THRESHOLD DECISION",
      },
    ],
  },
  conclusion: {
    headline: "Two models. One credit decision framework.",
    subline:
      "Combined, our regression and classification models give the credit desk data-backed answers at the point of decision. The models are complementary: income verification catches fabricated applications; risk scoring catches genuine but high-risk borrowers.",
    insights: [
      {
        title: "Step 1 — Income Plausibility Check",
        body: "Run the regression model on the applicant's profile. If declared income exceeds the model's 90th-percentile predicted range, flag for income document verification before proceeding to credit assessment. This step catches misrepresentation at the gate.",
        tag: "REGRESSION",
      },
      {
        title: "Step 2 — Repayment Risk Score",
        body: "Run the classification model. Probability above 0.70 → fast-track approval. Between 0.40–0.70 → credit analyst review. Below 0.40 → automatic decline or senior sign-off required. Thresholds are adjustable based on portfolio risk appetite.",
        tag: "CLASSIFICATION",
      },
      {
        title: "Step 3 — Risk Factor Explainability",
        body: "For borderline cases, surface the model's top contributing factors — prior defaults, debt-to-income ratio, interest rate tier — so analysts understand the score, not just accept it. This satisfies internal audit and regulatory explainability requirements.",
        tag: "EXPLAINABILITY",
      },
      {
        title: "Recommended next step — Production API",
        body: "Deploy both models as a real-time REST API endpoint. Each loan submission triggers both inferences synchronously. Credit desk receives the income flag, risk score, and top risk factors within seconds — before a human reviews the file.",
        tag: "DEPLOYMENT",
      },
    ],
    findings: [
      { label: "Income Model R²", value: "76.7%", note: "after log transformation" },
      { label: "Classifier AUC", value: "0.93", note: "excellent risk separation", highlight: true },
      { label: "Test Accuracy", value: "89.8%", note: "on 9,000 held-out records" },
      { label: "Key Risk Signal", value: "Prior Default", note: "dominant predictive factor" },
    ],
  },
};

export default function Home() {
  const [active, setActive] = useState("problem");
  const [copied, setCopied] = useState(false);

  const sec = content[active];

  const handleCopy = (code: string) => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  return (
    <div className="min-h-screen bg-[#08080f] text-[#e0dbd0] font-mono">
      <header className="border-b border-[#1a1a28] px-8 py-5 flex items-center justify-between sticky top-0 bg-[#08080f]/96 backdrop-blur z-40">
        <div>
          <div className="text-[9px] tracking-[0.5em] text-[#5a5a7a] uppercase mb-1">
            Credit Risk Intelligence · Loan Portfolio
          </div>
          <h1 className="text-lg font-bold tracking-tight">
            LENDING<span className="text-[#7c6fff]"> DECISION MODELS</span>
          </h1>
        </div>
        <div className="flex gap-8">
          {headerMetrics.map((m) => (
            <div key={m.label} className="text-right">
              <div className="text-xl font-bold text-[#7c6fff] tabular-nums">{m.value}</div>
              <div className="text-[8px] tracking-[0.3em] text-[#3a3a5a] uppercase">{m.label}</div>
            </div>
          ))}
        </div>
      </header>

      <div className="flex">
        <nav className="w-52 border-r border-[#1a1a28] min-h-[calc(100vh-69px)] p-3 shrink-0 sticky top-[69px] self-start">
          <div className="text-[8px] tracking-[0.4em] text-[#3a3a5a] uppercase mb-3 px-2">Workflow</div>
          {sections.map((s) => (
            <button
              key={s.id}
              onClick={() => setActive(s.id)}
              className={`w-full text-left px-3 py-2.5 rounded text-[12px] flex items-center gap-2 transition-all mb-0.5 ${
                active === s.id
                  ? "bg-[#7c6fff]/10 text-[#7c6fff] border border-[#7c6fff]/20"
                  : "text-[#5a5a7a] hover:text-[#e0dbd0] hover:bg-[#12121e]"
              }`}
            >
              <span className="text-[10px]">{s.icon}</span>
              {s.label}
            </button>
          ))}
          <div className="mt-6 px-2">
            <div className="text-[8px] tracking-[0.4em] text-[#3a3a5a] uppercase mb-2">Models Used</div>
            {[
              { name: "Linear Regression", tag: "INCOME" },
              { name: "Stepwise AIC", tag: "SELECTION" },
              { name: "Logistic Regression", tag: "RISK" },
            ].map((m) => (
              <div key={m.name} className="flex items-center justify-between py-1.5 border-b border-[#1a1a28]">
                <span className="text-[10px] text-[#4a4a6a]">{m.name}</span>
                <span className="text-[8px] text-[#7c6fff]">{m.tag}</span>
              </div>
            ))}
          </div>
          <div className="mt-4 px-2">
            <div className="text-[8px] tracking-[0.4em] text-[#3a3a5a] uppercase mb-2">Dataset</div>
            {[
              ["Records", "45,000"],
              ["Variables", "14"],
              ["Train split", "80%"],
              ["Test split", "20%"],
              ["Default rate", "~22.4%"],
            ].map(([k, v]) => (
              <div key={k} className="flex items-center justify-between py-1 border-b border-[#1a1a28]">
                <span className="text-[9px] text-[#3a3a5a]">{k}</span>
                <span className="text-[9px] text-[#6a6a9a]">{v}</span>
              </div>
            ))}
          </div>
        </nav>

        <main className="flex-1 p-8 max-w-3xl">
          <div className="text-[9px] tracking-[0.4em] text-[#3a3a5a] uppercase mb-2">
            {sections.find((s) => s.id === active)?.label}
          </div>
          <h2 className="text-2xl font-bold text-[#e0dbd0] mb-2 leading-tight">{sec.headline}</h2>
          <p className="text-[13px] text-[#6a6a8a] leading-relaxed mb-8">{sec.subline}</p>

          {sec.findings && (
            <div className={`grid gap-3 mb-8 ${sec.findings.length === 4 ? "grid-cols-4" : "grid-cols-3"}`}>
              {sec.findings.map((f) => (
                <div
                  key={f.label}
                  className={`border rounded-lg p-4 bg-[#0c0c18] ${
                    f.highlight ? "border-[#7c6fff]/40" : "border-[#1a1a28]"
                  }`}
                >
                  <div className={`text-xl font-bold tabular-nums ${f.highlight ? "text-[#7c6fff]" : "text-[#e0dbd0]"}`}>
                    {f.value}
                  </div>
                  <div className="text-[10px] text-[#8080a0] mt-1">{f.label}</div>
                  {f.note && <div className="text-[9px] text-[#3a3a5a] mt-0.5">{f.note}</div>}
                </div>
              ))}
            </div>
          )}

          {sec.metrics && (
            <div className="grid grid-cols-4 gap-3 mb-8">
              {sec.metrics.map((m) => (
                <div key={m.label} className="border border-[#1a1a28] rounded-lg p-4 bg-[#0c0c18]">
                  <div className="text-2xl font-bold tabular-nums" style={{ color: m.color ?? "#7c6fff" }}>
                    {m.val}
                  </div>
                  <div className="text-[10px] text-[#e0dbd0] mt-1">{m.label}</div>
                  <div className="text-[9px] text-[#3a3a5a] mt-0.5">{m.sub}</div>
                </div>
              ))}
            </div>
          )}

          {sec.insights && (
            <div className="space-y-3 mb-8">
              {sec.insights.map((ins, i) => (
                <div key={i} className="border border-[#1a1a28] rounded-lg p-5 bg-[#0a0a16]">
                  <div className="flex items-start justify-between gap-4 mb-2">
                    <h3 className="text-[13px] font-bold text-[#e0dbd0]">{ins.title}</h3>
                    {ins.tag && (
                      <span
                        className="text-[8px] tracking-[0.3em] font-bold px-2 py-0.5 rounded shrink-0"
                        style={{
                          color: tagColors[ins.tag] ?? "#7c6fff",
                          backgroundColor: (tagColors[ins.tag] ?? "#7c6fff") + "18",
                          border: `1px solid ${(tagColors[ins.tag] ?? "#7c6fff")}30`,
                        }}
                      >
                        {ins.tag}
                      </span>
                    )}
                  </div>
                  <p className="text-[12px] text-[#6a6a8a] leading-relaxed">{ins.body}</p>
                </div>
              ))}
            </div>
          )}

          {sec.codeBlock && (
            <div className="border border-[#1a1a28] rounded-lg overflow-hidden mb-8">
              <div className="flex items-center justify-between bg-[#0c0c18] px-4 py-2.5 border-b border-[#1a1a28]">
                <span className="text-[11px] text-[#7c6fff]">{sec.codeBlock.title}</span>
                <button
                  onClick={() => handleCopy(sec.codeBlock!.code)}
                  className="text-[9px] tracking-widest text-[#3a3a5a] hover:text-[#e0dbd0] transition-colors"
                >
                  {copied ? "✓ COPIED" : "COPY"}
                </button>
              </div>
              <div className="bg-[#0a0a16] p-5">
                <pre className="text-[12px] leading-relaxed text-[#c0bba8] whitespace-pre-wrap">
                  <code>{sec.codeBlock.code}</code>
                </pre>
              </div>
              <div className="bg-[#060610] border-t border-[#1a1a28] px-5 py-3">
                <div className="text-[8px] tracking-[0.4em] text-[#3a3a5a] uppercase mb-2">Result</div>
                <pre className="text-[11px] text-[#4a8e5f] leading-relaxed whitespace-pre-wrap">
                  {sec.codeBlock.output}
                </pre>
              </div>
            </div>
          )}

          {sec.matrix && (
            <div className="mb-8">
              <div className="text-[9px] tracking-[0.4em] text-[#3a3a5a] uppercase mb-3">
                Confusion Matrix — Test Set (9,000 records)
              </div>
              <div className="border border-[#1a1a28] rounded-lg overflow-hidden">
                <div className="grid grid-cols-3 text-[10px]">
                  <div className="bg-[#0a0a16] p-3 border-b border-r border-[#1a1a28] text-[#3a3a5a]" />
                  <div className="bg-[#0a0a16] p-3 border-b border-r border-[#1a1a28] text-center text-[#5a5a7a]">Predicted: Repay</div>
                  <div className="bg-[#0a0a16] p-3 border-b border-[#1a1a28] text-center text-[#5a5a7a]">Predicted: Default</div>
                  <div className="bg-[#0a0a16] p-3 border-r border-b border-[#1a1a28] text-[#5a5a7a]">Actual: Repay</div>
                  <div className="bg-[#0d1a12] p-4 border-r border-b border-[#1a1a28] text-center">
                    <div className="text-xl font-bold text-[#4a8e5f]">{sec.matrix.tn.toLocaleString()}</div>
                    <div className="text-[9px] text-[#4a8e5f] mt-1">True Negative</div>
                    <div className="text-[8px] text-[#3a3a5a]">correctly approved</div>
                  </div>
                  <div className="bg-[#1a0d0d] p-4 border-b border-[#1a1a28] text-center">
                    <div className="text-xl font-bold text-[#e05252]">{sec.matrix.fp.toLocaleString()}</div>
                    <div className="text-[9px] text-[#e05252] mt-1">False Positive</div>
                    <div className="text-[8px] text-[#3a3a5a]">flagged for review</div>
                  </div>
                  <div className="bg-[#0a0a16] p-3 border-r border-[#1a1a28] text-[#5a5a7a]">Actual: Default</div>
                  <div className="bg-[#1a150d] p-4 border-r border-[#1a1a28] text-center">
                    <div className="text-xl font-bold text-[#d4a843]">{sec.matrix.fn.toLocaleString()}</div>
                    <div className="text-[9px] text-[#d4a843] mt-1">False Negative</div>
                    <div className="text-[8px] text-[#3a3a5a]">missed defaults</div>
                  </div>
                  <div className="bg-[#0d1a12] p-4 text-center">
                    <div className="text-xl font-bold text-[#4a8e5f]">{sec.matrix.tp.toLocaleString()}</div>
                    <div className="text-[9px] text-[#4a8e5f] mt-1">True Positive</div>
                    <div className="text-[8px] text-[#3a3a5a]">correctly flagged</div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {sec.riskFactors && (
            <div className="mb-8">
              <div className="text-[9px] tracking-[0.4em] text-[#3a3a5a] uppercase mb-3">
                Risk Factor Hierarchy — By Predictive Weight
              </div>
              <div className="border border-[#1a1a28] rounded-lg overflow-hidden">
                {sec.riskFactors.map((r, i) => (
                  <div
                    key={i}
                    className="flex items-center justify-between px-5 py-3 border-b border-[#1a1a28] last:border-0 hover:bg-[#0c0c18] transition-colors"
                  >
                    <span className="text-[12px] text-[#c0bba8]">{r.factor}</span>
                    <span
                      className="text-[9px] font-bold tracking-[0.2em] px-2 py-1 rounded"
                      style={{
                        color: r.color,
                        backgroundColor: r.color + "15",
                        border: `1px solid ${r.color}30`,
                      }}
                    >
                      {r.level}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
