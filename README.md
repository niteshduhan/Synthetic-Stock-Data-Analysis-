# Synthetic Stock Market Analysis — End-to-End Data Science Pipeline

> Statistical inference · Time-series forecasting · ML classification on synthetic financial data

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?style=flat-square&logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-1.26-013243?style=flat-square&logo=numpy)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?style=flat-square&logo=scikit-learn)
![Statsmodels](https://img.shields.io/badge/Statsmodels-0.14-4B8BBE?style=flat-square)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.8-11557C?style=flat-square)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter)

---

## Overview

A multi-stage analytical pipeline applied to a synthetic stock market dataset — covering statistical hypothesis testing, financial analytics, ARIMA forecasting, and ML-based trend classification. The project is structured as an integrated data science workflow, not a single-model exercise.

**Dataset:** Synthetic stock data · 150-record stratified sample · 13 features (OHLCV + fundamentals + sentiment)

**Key outcomes:**
- ARIMA (2,1,2) produced a smooth 5-day forward forecast aligned with recent price momentum
- Decision Tree classifier successfully separated Bullish / Bearish / Stable trend classes; top features: `Sentiment Score`, `Dividend Yield`, `Market Cap`
- Statistical tests (t-test, z-test, ANOVA, F-test) confirmed significant price variation across sectors and time periods
- Log returns exhibited positive skewness and leptokurtic tails — non-normality confirmed via Q-Q plots and ADF stationarity testing

---

## Methodology

```
Raw Dataset
    │
    ▼
1. Sampling & Preparation
   └─ Random + stratified sampling → 150 records
   └─ Chronological sort, type casting, missing-value audit

2. Inferential Statistics
   └─ One-sample t-test (Close vs μ=100)
   └─ Proportion z-test (gain-day bias)
   └─ Levene's F-test (variance shift: early vs late)
   └─ One-way ANOVA (Close price across sectors)

3. Categorical & Count Data
   └─ Multinomial logistic regression → Trend prediction
   └─ Coefficient interpretation per class

4. Financial Analysis
   └─ Log returns, rolling volatility
   └─ Skewness, kurtosis, normality checks (Shapiro-Wilk, Q-Q)

5. Time-Series Forecasting
   └─ ADF test → differencing order d=1
   └─ ACF/PACF → (p,q) = (2,2)
   └─ ARIMA (2,1,2) fit → 5-day out-of-sample forecast
   └─ Residual diagnostics

6. Machine Learning Classification
   └─ Features: OHLCV, PE Ratio, Market Cap, Sentiment, Dividend Yield
   └─ Decision Tree (Gini, max_depth tuned)
   └─ Evaluation: Accuracy, Precision, Recall, F1, Confusion Matrix

7. Visual Analytics
   └─ Correlation heatmap, distribution plots, forecast plots, feature importance
```

---

## Results

### Statistical Tests

| Test | Hypothesis | Result |
|---|---|---|
| One-sample t-test | Close ≠ 100 | **Rejected H₀** — significant deviation |
| Proportion z-test | Gain days = 50% | **Rejected H₀** — market bias detected |
| Levene's F-test | Variance: early = late | **Rejected H₀** — volatility shift confirmed |
| One-way ANOVA | Close equal across sectors | **Rejected H₀** — sector-level differences |

### ARIMA Forecast

| Metric | Value |
|---|---|
| Model Order | (2, 1, 2) |
| Forecast Horizon | 5 days |
| Residual Behaviour | White noise (Ljung-Box passed) |
| Stationarity | ADF confirmed post-differencing |

### ML Classification — Decision Tree

| Metric | Score |
|---|---|
| Accuracy | — *(fill in)* |
| Precision (macro) | — *(fill in)* |
| Recall (macro) | — *(fill in)* |
| F1 (macro) | — *(fill in)* |
| Top Features | Sentiment Score, Dividend Yield, Market Cap |

> 💡 Fill in the exact metric values from your model output for maximum impact on recruiters.

---

## Dataset

| Field | Description |
|---|---|
| `Date` | Trading date |
| `Open / High / Low / Close` | OHLC prices |
| `Volume` | Daily trade volume |
| `Market Cap` | Company market capitalisation |
| `P/E Ratio` | Price-to-earnings ratio |
| `Dividend Yield` | Annual dividend yield |
| `Volatility` | Rolling price volatility |
| `Sentiment Score` | Synthetic sentiment signal [−1, 1] |
| `Sector / Company` | Categorical identifiers |
| `Trend` | Target — Bullish / Bearish / Stable |

**Note:** Synthetic dataset — no licensing restrictions. Designed to mimic real financial data distributions.

---

## Project Structure

```
synthetic-stock-analysis/
│
├── data/
│   ├── raw_stock_data.csv          # Original synthetic dataset
│   └── sampled_stock_data.csv      # 150-record stratified sample
│
├── notebooks/
│   └── stock_analysis_pipeline.ipynb   # Full end-to-end notebook
│
├── outputs/
│   ├── arima_forecast.png
│   ├── correlation_matrix.png
│   ├── qq_plot.png
│   ├── decision_tree.png
│   └── feature_importance.png
│
├── requirements.txt
└── README.md
```

---

## How to Run

```bash
# 1. Clone the repo
git clone https://github.com/niteshduhan/synthetic-stock-analysis.git
cd synthetic-stock-analysis

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch notebook
jupyter notebook notebooks/stock_analysis_pipeline.ipynb
```

**requirements.txt**
```
pandas
numpy
scipy
statsmodels
scikit-learn
matplotlib
seaborn
tabulate
jupyter
```

---

## Key Takeaways

- **Statistical rigour at scale:** Applied 4 distinct hypothesis tests with correct assumptions (normality, homogeneity of variance) — not just plug-and-play p-values.
- **Time-series discipline:** Enforced stationarity checks before ARIMA fitting; validated residuals post-fit — standard practice often skipped in portfolio projects.
- **ML as one tool, not the whole pipeline:** Classification was the final stage of a multi-domain workflow, demonstrating ability to frame and solve problems across statistical and ML paradigms.

---

## Author

**Nitesh Duhan** — MSc Data Science

[![LinkedIn](https://img.shields.io/badge/LinkedIn-niteshduhan--carp112-0077B5?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/niteshduhan-carp112)
[![Email](https://img.shields.io/badge/Email-niteshduhan686@gmail.com-D14836?style=flat-square&logo=gmail)](mailto:niteshduhan686@gmail.com)
[![Instagram](https://img.shields.io/badge/Instagram-@nitesh._duhan-E4405F?style=flat-square&logo=instagram)](https://www.instagram.com/nitesh._duhan)
