# 📈 Stock Price Predictor — Machine Learning

A Python machine learning project that pulls real stock market data and predicts the next day's closing price using engineered financial features and two trained models. Built with `yfinance`, `scikit-learn`, and `matplotlib`.

---

## Overview

This project fetches historical stock data directly from Yahoo Finance and uses it to train two machine learning models — Linear Regression and Gradient Boosting — to predict the next day's closing price. It evaluates both models against real test data, picks the best performer, and outputs a next-day price prediction along with a buy/sell signal.

The default ticker is **AAPL (Apple Inc.)**, but any valid stock ticker can be used by changing a single variable at the top of the script.

---

## How It Works

### 1. Data Collection
Historical OHLCV data (Open, High, Low, Close, Volume) is fetched directly from Yahoo Finance using the `yfinance` library for a configurable date range.

### 2. Feature Engineering
Raw price data alone is not enough for a good model. The script engineers 15 technical indicators commonly used in quantitative finance:

| Feature | Description |
|---|---|
| MA_10 / MA_30 / MA_60 | Simple moving averages over 10, 30, 60 days |
| EMA_12 / EMA_26 | Exponential moving averages |
| MACD | Moving Average Convergence Divergence (EMA_12 − EMA_26) |
| RSI | Relative Strength Index (14-day) — momentum oscillator |
| Volatility | Rolling 10-day standard deviation of closing price |
| Daily_Return | Day-over-day percentage change |
| Momentum | Price difference over 10-day window |
| Price_Range | Difference between daily High and Low |
| Volume_MA | 10-day rolling average of trading volume |

### 3. Model Training
Data is split 80/20 into training and test sets. Features are scaled using MinMaxScaler before training.

- **Linear Regression** — fast, interpretable baseline model
- **Gradient Boosting** — ensemble model that builds trees sequentially, correcting previous errors, resulting in stronger predictions on complex non-linear patterns

### 4. Evaluation
Both models are evaluated on the unseen test set using four metrics:

- **MAE** (Mean Absolute Error) — average dollar error
- **RMSE** (Root Mean Squared Error) — penalises large errors more heavily
- **R²** (R-squared) — percentage of price variance explained by the model
- **MAPE** (Mean Absolute Percentage Error) — error as a percentage of actual price

### 5. Prediction
The best-performing model predicts tomorrow's closing price based on the most recent available data, outputting an expected price, dollar change, percentage change, and a buy/sell signal.

> This project is built for **educational purposes only**. It is not financial advice. Past stock performance does not guarantee future results. Do not make real investment decisions based on this model.
