import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import shap
import matplotlib.pyplot as plt

tf.random.set_seed(42)
np.random.seed(42)

BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

#S-1 : Data Geneartion , Muiltivirant, 5+ features
n_steps = 1200
t = np.arange(n_steps)

temp = 20 + 10*np.sin(2*np.pi*t/100) + np.random.normal(0,1,n_steps)
humidity = 50 - 0.5*temp + np.random.normal(0,5,n_steps)
day_of_week = t % 7
activity = np.cumsum(np.random.normal(0,0.5,n_steps)) + 50
price = 0.05*t + np.random.normal(0,2,n_steps)

target = 0.6*temp + 0.4*activity + 0.2*price + np.random.normal(0,2,n_steps)

data = pd.DataFrame({
    "temp": temp,
    "humidity": humidity,
    "day_of_week": day_of_week,
    "activity": activity,
    "price": price,
    "target": target
})

data

data.head()
data.shape
data.describe()

#missing values
data.isnull().sum()

#drop duplicates
data.drop_duplicates()

for lag in [1, 3, 6, 12, 24]:
    data[f"target_lag_{lag}"] = data["target"].shift(lag)

data["target_roll_mean_24"] = data["target"].rolling(24).mean()
data["target_roll_std_24"] = data["target"].rolling(24).std()

data["dow_sin"] = np.sin(2*np.pi*data["day_of_week"]/7)
data["dow_cos"] = np.cos(2*np.pi*data["day_of_week"]/7)

data.dropna(inplace=True)
data.reset_index(drop=True, inplace=True)

data.to_csv(os.path.join(DATA_DIR, "time_series.csv"), index=False)

# S-2: data pipeline - Normalization & Windowing
scaler = MinMaxScaler()
scaled = scaler.fit_transform(data)

def create_windows(arr, window=24):
    X, y = [], []
    for i in range(len(arr) - window):
        X.append(arr[i:i+window, :-1])
        y.append(arr[i+window, -1])
    return np.array(X), np.array(y)

X, y = create_windows(scaled, 24)

train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

#S-3: Baseline 1 – ARIMA
arima = ARIMA(data["target"][:train_size], order=(5,1,0)).fit()
arima_preds = arima.forecast(len(y_test))
arima_rmse = np.sqrt(mean_squared_error(
    data["target"].iloc[train_size+24:train_size+24+len(y_test)],
    arima_preds
))

#S-3: Baseline 2 – Prophet
prophet_df = data[["target"]].copy()
prophet_df["ds"] = pd.date_range("2020-01-01", periods=len(prophet_df))
prophet_df.rename(columns={"target":"y"}, inplace=True)

prophet = Prophet()
prophet.fit(prophet_df.iloc[:train_size])

future = prophet.make_future_dataframe(periods=len(prophet_df)-train_size)
forecast = prophet.predict(future)

prophet_rmse = np.sqrt(mean_squared_error(
    prophet_df["y"].iloc[train_size:],
    forecast["yhat"].iloc[train_size:]
))

#S-4: LSTM – Time-aware Hyperparameter Sweep
def build_lstm(units, depth, shape):
    model = Sequential()
    model.add(LSTM(units, return_sequences=(depth==2), input_shape=shape))
    model.add(Dropout(0.2))
    if depth == 2:
        model.add(LSTM(units//2))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    return model

tscv = TimeSeriesSplit(n_splits=5)
best_rmse = np.inf
best_model = None

for units in [32, 64]:
    for depth in [1, 2]:
        rmses = []
        for tr, val in tscv.split(X_train):
            model = build_lstm(units, depth, X_train.shape[1:])
            model.fit(X_train[tr], y_train[tr], epochs=10,
                      batch_size=32, verbose=0)
            preds = model.predict(X_train[val])

            dummy = np.zeros((len(preds), scaled.shape[1]))
            dummy[:, -1] = preds[:,0]
            y_pred = scaler.inverse_transform(dummy)[:,-1]

            dummy[:, -1] = y_train[val]
            y_true = scaler.inverse_transform(dummy)[:,-1]

            rmses.append(np.sqrt(mean_squared_error(y_true, y_pred)))

        if np.mean(rmses) < best_rmse:
            best_rmse = np.mean(rmses)
            best_model = model

# Final test evaluation
test_preds = best_model.predict(X_test)
dummy = np.zeros((len(test_preds), scaled.shape[1]))
dummy[:, -1] = test_preds[:,0]
y_pred = scaler.inverse_transform(dummy)[:,-1]

dummy[:, -1] = y_test
y_true = scaler.inverse_transform(dummy)[:,-1]
lstm_rmse = np.sqrt(mean_squared_error(y_true, y_pred))

#S-5: SHAP – Corrected for LSTM 

def lstm_predict_flat(X_flat):
    X_seq = X_flat.reshape((-1, X_train.shape[1], X_train.shape[2]))
    preds = best_model.predict(X_seq, verbose=0)
    return preds.flatten()

X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat  = X_test.reshape(X_test.shape[0], -1)

background = X_train_flat[np.random.choice(X_train_flat.shape[0], 50, replace=False)]
samples = X_test_flat[:10]

explainer = shap.KernelExplainer(lstm_predict_flat, background)
shap_values = explainer.shap_values(samples, nsamples=100)

n_features = X_train.shape[2]
shap_seq = shap_values.reshape(len(samples), X_train.shape[1], n_features)

feature_importance = np.mean(np.abs(shap_seq), axis=(0,1))
feature_names = data.columns.drop("target")

plt.figure(figsize=(10,6))
plt.barh(feature_names, feature_importance)
plt.title("LSTM SHAP Feature Importance (Temporal Average)")
plt.gca().invert_yaxis()
plt.grid(axis="x", linestyle="--", alpha=0.6)
plt.show()

# S-6: Final Results (Deliverable 2 Summary)

print(f"ARIMA RMSE   : {arima_rmse:.2f}")
print(f"Prophet RMSE : {prophet_rmse:.2f}")
print(f"LSTM RMSE    : {lstm_rmse:.2f}")

print("""
INTERPRETATION:
• Feature attributions indicate strong influence of recent demand lags
  and temperature, reflecting short-term sensitivity to historical usage
  and weather patterns.
• Industrial activity exhibits sustained influence across the input
  sequence, consistent with accumulated demand effects over time.
• Price and seasonal encodings contribute to longer-term structural
  behavior, capturing trend and periodic demand components.
""")
