import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")


TICKER = "AAPL"
START_DATE = "2020-01-01"
END_DATE = "2024-12-31"
PREDICTION_DAYS = 30
FEATURE_WINDOW = 60


def fetch_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False)
    df = df[["Close", "Volume", "High", "Low", "Open"]].copy()
    df.dropna(inplace=True)
    return df


def add_features(df):
    df = df.copy()
    df["MA_10"]       = df["Close"].rolling(10).mean()
    df["MA_30"]       = df["Close"].rolling(30).mean()
    df["MA_60"]       = df["Close"].rolling(60).mean()
    df["EMA_12"]      = df["Close"].ewm(span=12).mean()
    df["EMA_26"]      = df["Close"].ewm(span=26).mean()
    df["MACD"]        = df["EMA_12"] - df["EMA_26"]
    df["Volatility"]  = df["Close"].rolling(10).std()
    df["Daily_Return"]= df["Close"].pct_change()
    df["Momentum"]    = df["Close"] - df["Close"].shift(10)
    df["Price_Range"] = df["High"] - df["Low"]
    delta             = df["Close"].diff()
    gain              = delta.clip(lower=0).rolling(14).mean()
    loss              = (-delta.clip(upper=0)).rolling(14).mean()
    rs                = gain / loss
    df["RSI"]         = 100 - (100 / (1 + rs))
    df["Volume_MA"]   = df["Volume"].rolling(10).mean()
    df["Target"]      = df["Close"].shift(-1)
    df.dropna(inplace=True)
    return df


def build_features_target(df):
    feature_cols = [
        "Close", "Volume", "High", "Low", "Open",
        "MA_10", "MA_30", "MA_60", "MACD",
        "Volatility", "Daily_Return", "Momentum",
        "Price_Range", "RSI", "Volume_MA"
    ]
    X = df[feature_cols].values
    y = df["Target"].values
    return X, y, feature_cols


def evaluate(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(f"\n  {name}")
    print(f"  {'─'*30}")
    print(f"  MAE:   ${mae:.2f}")
    print(f"  RMSE:  ${rmse:.2f}")
    print(f"  R²:    {r2:.4f}")
    print(f"  MAPE:  {mape:.2f}%")
    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape}


print(f"📈 Fetching {TICKER} data from {START_DATE} to {END_DATE}...")
raw_df = fetch_data(TICKER, START_DATE, END_DATE)
print(f"✅ {len(raw_df)} trading days loaded")

df = add_features(raw_df)

X, y, feature_cols = build_features_target(df)

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
test_dates = df.index[split:]

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled  = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

print(f"\nTraining samples: {len(X_train)} | Test samples: {len(X_test)}")

lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train_scaled)
lr_preds = scaler_y.inverse_transform(lr_model.predict(X_test_scaled).reshape(-1, 1)).ravel()

gb_model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    random_state=42
)
gb_model.fit(X_train_scaled, y_train_scaled)
gb_preds = scaler_y.inverse_transform(gb_model.predict(X_test_scaled).reshape(-1, 1)).ravel()

print(f"\n{'='*45}")
print(f"  MODEL EVALUATION — {TICKER}")
print(f"{'='*45}")
lr_scores = evaluate("Linear Regression", y_test, lr_preds)
gb_scores = evaluate("Gradient Boosting", y_test, gb_preds)

best_preds = gb_preds if gb_scores["rmse"] < lr_scores["rmse"] else lr_preds
best_name  = "Gradient Boosting" if gb_scores["rmse"] < lr_scores["rmse"] else "Linear Regression"
print(f"\n  🏆 Best model: {best_name}")

last_features = X[-1].reshape(1, -1)
last_scaled   = scaler_X.transform(last_features)
next_price    = scaler_y.inverse_transform(
    gb_model.predict(last_scaled).reshape(-1, 1)
).ravel()[0]
current_price = float(df["Close"].iloc[-1])
change        = next_price - current_price
change_pct    = (change / current_price) * 100

print(f"\n{'='*45}")
print(f"  NEXT DAY PREDICTION")
print(f"{'='*45}")
print(f"  Current price:    ${current_price:.2f}")
print(f"  Predicted price:  ${next_price:.2f}")
print(f"  Expected change:  {'+' if change >= 0 else ''}{change:.2f} ({change_pct:+.2f}%)")
print(f"  Signal:           {'📈 BUY' if change > 0 else '📉 SELL'}")

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle(f"{TICKER} Stock Price Prediction — ML Analysis", fontsize=16, fontweight="bold")

ax1 = axes[0, 0]
ax1.plot(df.index, df["Close"], color="#1f77b4", linewidth=1.2, label="Actual Price")
ax1.axvline(test_dates[0], color="red", linestyle="--", alpha=0.7, label="Train/Test Split")
ax1.set_title("Full Price History")
ax1.set_ylabel("Price ($)")
ax1.legend()
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

ax2 = axes[0, 1]
ax2.plot(test_dates, y_test,    color="#1f77b4", linewidth=1.2, label="Actual",             alpha=0.8)
ax2.plot(test_dates, lr_preds,  color="#ff7f0e", linewidth=1.0, label="Linear Regression",  alpha=0.8)
ax2.plot(test_dates, gb_preds,  color="#2ca02c", linewidth=1.0, label="Gradient Boosting",  alpha=0.8)
ax2.set_title("Test Set: Predicted vs Actual")
ax2.set_ylabel("Price ($)")
ax2.legend()
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

ax3 = axes[1, 0]
importances  = gb_model.feature_importances_
sorted_idx   = np.argsort(importances)[::-1]
sorted_names = [feature_cols[i] for i in sorted_idx]
colors = ["#2ca02c" if i == 0 else "#1f77b4" for i in range(len(importances))]
ax3.bar(range(len(importances)), importances[sorted_idx], color=colors)
ax3.set_xticks(range(len(importances)))
ax3.set_xticklabels(sorted_names, rotation=45, ha="right")
ax3.set_title("Feature Importances (Gradient Boosting)")
ax3.set_ylabel("Importance")

ax4 = axes[1, 1]
residuals = y_test - gb_preds
ax4.scatter(gb_preds, residuals, alpha=0.4, s=10, color="#2ca02c")
ax4.axhline(0, color="red", linestyle="--", linewidth=1.5)
ax4.set_xlabel("Predicted Price ($)")
ax4.set_ylabel("Residual ($)")
ax4.set_title("Residual Plot (Gradient Boosting)")

plt.tight_layout()
plt.savefig("stock_prediction.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n Chart saved as stock_prediction.png")

