# 📈 Synthetic Stock Data Analysis — Integrated Data Science & Machine Learning Project

This project presents a comprehensive end-to-end analysis of synthetic stock market data using a blend of statistical methods, time-series modeling, sampling techniques, and machine learning algorithms. It demonstrates a complete data science workflow — from sampling and preprocessing to hypothesis testing, forecasting, classification, visualization, and interpretation.

# 🚀 Project Overview

The objective of this project is to perform a multi-stage analytical exploration of stock market data by applying core concepts from statistics, time-series forecasting, financial analytics, and machine learning.
The project integrates multiple domains — Sampling Techniques, Inferential Statistics, Count & Categorical Data Analysis, Financial Data Analysis, ARIMA Time Series Modeling, and Machine Learning Classification.

This project follows a structured workflow:

Sampling & Data Preparation: Random sampling, chronological sorting, preprocessing.

Inferential Statistics: t-tests, proportion tests, F-tests, and ANOVA to validate hypotheses.

Categorical & Count Data Analysis: Logistic regression for multiclass trend prediction.

Financial Data Analysis: Log returns, volatility, descriptive statistics, normality checks.

Time Series Modeling: ARIMA fitting, model diagnostics, and 5-day forecasting.

Machine Learning: Decision Tree classifier, feature analysis, and classification metrics.

Visual Analytics: Line plots, scatter plots, distribution plots, Q-Q plots, correlation matrix.

# 📂 Key Features

Chronologically prepared synthetic financial dataset

Random and stratified sampling for representative subsets

Full inferential statistical pipeline (t-test, z-test, ANOVA, F-test)

Categorical modeling using logistic regression

Financial analysis including log returns, volatility, skewness, and kurtosis

Normality assessment using distribution plots and Q-Q plots

Time-series forecasting with ARIMA (2,1,2)

Machine learning classification using Decision Trees

Detailed visualizations: heatmaps, plots, trends, and forecasts

Interpretation-focused explanations for each analytical step

# 🧰 Tech Stack

Python

Pandas, NumPy, SciPy, Statsmodels

Matplotlib, Seaborn

Scikit-Learn (Logistic Regression, Decision Tree)

Statsmodels (ADF Test, ANOVA, ARIMA)

Tabulate, IPython.display

Jupyter Notebook / VS Code

# 📊 Results & Insights

Sampling successfully produced a clean, chronologically ordered subset of 150 stock records, enabling reliable time-series and statistical analysis.

Inferential Statistics (t-test, proportion z-test, F-test, ANOVA) revealed:

The average Close price differs significantly from 100.

The proportion of gain days is not equal to 50%, indicating market bias.

Variance between early and late periods shows volatility changes over time.

Closing prices vary significantly across sectors.

Categorical Analysis using Logistic Regression produced clear trend classifications with interpretable coefficients, highlighting which financial features influence market direction.

Financial Data Analysis showed log returns centered around zero with positive skewness and fat tails, confirming non-normal behavior (supported by the distribution & Q-Q plot on pages 14–15).

ARIMA (2,1,2) effectively captured closing-price trends and produced a smooth 5-day forecast closely aligned with recent price movements (page 20).

Decision Tree Classifier successfully classified Bullish/Bearish/Stable trends and highlighted influential features such as Company, Dividend Yield, Sentiment Score, and Market Cap.

Correlation Matrix (page 23) showed extremely strong internal correlation among Open–High–Low–Close, and weak correlations with Volume, PE Ratio, and Market Cap.

# 📁 Dataset

The project uses a synthetic stock dataset containing fields such as:

Date, Open, High, Low, Close

Volume

Market Cap

P/E Ratio

Dividend Yield

Volatility

Sentiment Score

Sector, Company, Trend

This dataset is sampled, cleaned, encoded, and transformed for statistical and ML workflows.

# 🤝 Contributions

Suggestions, improvements, and extensions (such as adding RNN models, sentiment pipelines, or risk metrics) are welcome.
Feel free to open issues or submit pull requests.
