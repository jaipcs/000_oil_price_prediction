import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================
# PAGE SETTINGS
# =========================
st.set_page_config(page_title="Crude Oil Forecast Dashboard", layout="wide")

st.title("Crude Oil Price Forecasting Dashboard")
st.write("Compare models, inspect data, predict crude oil close price, and forecast future values.")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    possible_paths = [
        "crude_oil_final.csv",
        "./crude_oil_final.csv",
        "data/crude_oil_final.csv",
        "./data/crude_oil_final.csv"
    ]

    for path in possible_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date").reset_index(drop=True)
            return df

    st.error("CSV file not found. Upload crude_oil_final.csv to your project folder.")
    st.stop()

df = load_data()

# =========================
# EXTRA FEATURES
# =========================
df["Return_1"] = df["Close"].pct_change()
df["Momentum_3"] = df["Close"] - df["Close"].shift(3)
df["Momentum_7"] = df["Close"] - df["Close"].shift(7)
df["Close_roll_mean_14"] = df["Close"].rolling(14).mean()
df["Trend_7_14"] = df["Close_roll_mean_7"] - df["Close_roll_mean_14"]

# =========================
# FEATURE SET
# =========================
target_col = "Close"

feature_cols = [
    "Close_lag_1",
    "Close_lag_3",
    "Close_lag_7",
    "Close_roll_mean_7",
    "Volatility_7",
    "Return_1",
    "Momentum_3",
    "Momentum_7",
    "Trend_7_14"
]

feature_cols = [col for col in feature_cols if col in df.columns]

if len(feature_cols) == 0:
    st.error("Required feature columns were not found.")
    st.stop()

df_model = df[["Date", target_col] + feature_cols].dropna().copy()

# =========================
# TRAIN / TEST SPLIT
# =========================
split_index = int(len(df_model) * 0.8)

train_df = df_model.iloc[:split_index].copy()
test_df = df_model.iloc[split_index:].copy()

X_train = train_df[feature_cols]
X_test = test_df[feature_cols]

y_train = train_df[target_col]
y_test = test_df[target_col]

dates_test = test_df["Date"]

# =========================
# METRICS
# =========================
def get_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    y_true_safe = np.where(np.array(y_true) == 0, 1e-8, np.array(y_true))
    mape = np.mean(np.abs((np.array(y_true) - np.array(y_pred)) / y_true_safe)) * 100
    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2
    }

# =========================
# TRAIN MODELS
# =========================
@st.cache_resource
def train_models(X_train, y_train):
    rf_model = RandomForestRegressor(
        n_estimators=400,
        max_depth=12,
        random_state=42
    )
    rf_model.fit(X_train, y_train)

    xgb_model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        objective="reg:squarederror"
    )
    xgb_model.fit(X_train, y_train)

    return rf_model, xgb_model

rf_model, xgb_model = train_models(X_train, y_train)

rf_pred = pd.Series(rf_model.predict(X_test), index=y_test.index)
xgb_pred = pd.Series(xgb_model.predict(X_test), index=y_test.index)

rf_metrics = get_metrics(y_test, rf_pred)
xgb_metrics = get_metrics(y_test, xgb_pred)

comparison_df = pd.DataFrame([
    {"Model": "Random Forest", **rf_metrics},
    {"Model": "XGBoost", **xgb_metrics}
]).sort_values("RMSE").reset_index(drop=True)

# =========================
# SIDEBAR
# =========================
st.sidebar.header("Controls")

selected_model = st.sidebar.selectbox(
    "Select model",
    ["Random Forest", "XGBoost"]
)

extra_future_days = st.sidebar.slider(
    "Extra future days after today",
    min_value=3,
    max_value=60,
    value=14,
    step=1
)

show_raw_data = st.sidebar.checkbox("Show raw dataset", value=False)
show_summary = st.sidebar.checkbox("Show summary statistics", value=True)

if selected_model == "Random Forest":
    model = rf_model
    y_pred = rf_pred
    model_metrics = rf_metrics
else:
    model = xgb_model
    y_pred = xgb_pred
    model_metrics = xgb_metrics

# =========================
# TOP METRICS
# =========================
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Dataset Rows", len(df_model))

with col2:
    st.metric("Train Rows", len(train_df))

with col3:
    st.metric("Test Rows", len(test_df))

# =========================
# DATA PREVIEW
# =========================
st.subheader("Dataset Preview")

if show_raw_data:
    st.dataframe(df_model)
else:
    st.dataframe(df_model.head())

if show_summary:
    st.subheader("Summary Statistics")
    st.dataframe(df_model.describe())

# =========================
# HISTORICAL CLOSE
# =========================
st.subheader("Historical Close Price")

fig1, ax1 = plt.subplots(figsize=(12, 4))
ax1.plot(df_model["Date"], df_model["Close"], linewidth=2)
ax1.set_title("Crude Oil Close Price Over Time")
ax1.set_xlabel("Date")
ax1.set_ylabel("Close Price")
ax1.grid(True)
st.pyplot(fig1)

# =========================
# MODEL PERFORMANCE
# =========================
st.subheader("Model Performance Comparison")
st.dataframe(comparison_df)

st.subheader(f"Selected Model: {selected_model}")

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric("RMSE", f"{model_metrics['RMSE']:.3f}")

with m2:
    st.metric("MAE", f"{model_metrics['MAE']:.3f}")

with m3:
    st.metric("MAPE", f"{model_metrics['MAPE']:.3f}%")

with m4:
    st.metric("R²", f"{model_metrics['R2']:.3f}")

# =========================
# ACTUAL VS PREDICTED
# =========================
st.subheader(f"Actual vs Predicted - {selected_model}")

fig2, ax2 = plt.subplots(figsize=(12, 5))
ax2.plot(dates_test, y_test.values, label="Actual Close", linewidth=3)
ax2.plot(dates_test, y_pred.values, label=f"Predicted - {selected_model}", linewidth=2)
ax2.set_title(f"Actual vs Predicted Close - {selected_model}")
ax2.set_xlabel("Date")
ax2.set_ylabel("Close Price")
ax2.legend()
ax2.grid(True)
st.pyplot(fig2)

# =========================
# ERROR PLOT
# =========================
st.subheader(f"Prediction Error - {selected_model}")

error_values = y_test.values - y_pred.values

fig3, ax3 = plt.subplots(figsize=(12, 4))
ax3.plot(dates_test, error_values, linewidth=2)
ax3.axhline(0, linestyle="--")
ax3.set_title(f"Prediction Error - {selected_model}")
ax3.set_xlabel("Date")
ax3.set_ylabel("Error")
ax3.grid(True)
st.pyplot(fig3)

# =========================
# FEATURE IMPORTANCE
# =========================
st.subheader(f"Feature Importance - {selected_model}")

if hasattr(model, "feature_importances_"):
    importance_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    fig4, ax4 = plt.subplots(figsize=(10, 4))
    ax4.bar(importance_df["Feature"], importance_df["Importance"])
    ax4.set_title(f"Feature Importance - {selected_model}")
    ax4.set_xlabel("Feature")
    ax4.set_ylabel("Importance")
    ax4.tick_params(axis="x", rotation=45)
    st.pyplot(fig4)

    st.dataframe(importance_df)

# =========================
# MANUAL PREDICTION
# =========================
st.subheader("Manual Prediction")

input_values = []

input_col1, input_col2 = st.columns(2)

for i, col in enumerate(feature_cols):
    default_value = float(df_model[col].iloc[-1])

    if i % 2 == 0:
        with input_col1:
            val = st.number_input(col, value=default_value, format="%.6f")
    else:
        with input_col2:
            val = st.number_input(col, value=default_value, format="%.6f")

    input_values.append(val)

input_array = pd.DataFrame([input_values], columns=feature_cols)

if st.button("Predict Close Price"):
    prediction = model.predict(input_array)[0]
    st.success(f"Predicted Close Price using {selected_model}: {prediction:.2f}")

# =========================
# FIXED FUTURE FORECAST FUNCTION
# =========================
st.subheader("Future Close Price Forecast")

def recursive_future_forecast_until_today(model, df_model, feature_cols, extra_future_days):
    last_known_date = df_model["Date"].iloc[-1]
    today_date = pd.Timestamp.today().normalize()

    if last_known_date >= today_date:
        total_days = extra_future_days
    else:
        days_until_today = (today_date - last_known_date).days
        total_days = days_until_today + extra_future_days

    last_close_values = list(df_model["Close"].tail(30).values)

    future_predictions = []

    recent_returns = df_model["Close"].pct_change().dropna().tail(30)
    avg_return = recent_returns.mean()
    return_std = recent_returns.std()

    np.random.seed(42)

    for day in range(1, total_days + 1):

        close_lag_1 = last_close_values[-1]
        close_lag_3 = last_close_values[-3]
        close_lag_7 = last_close_values[-7]

        close_roll_mean_7 = np.mean(last_close_values[-7:])
        volatility_7 = np.std(last_close_values[-7:])

        return_1 = (last_close_values[-1] - last_close_values[-2]) / last_close_values[-2]
        momentum_3 = last_close_values[-1] - last_close_values[-3]
        momentum_7 = last_close_values[-1] - last_close_values[-7]

        close_roll_mean_14 = np.mean(last_close_values[-14:])
        trend_7_14 = close_roll_mean_7 - close_roll_mean_14

        feature_dict = {
            "Close_lag_1": close_lag_1,
            "Close_lag_3": close_lag_3,
            "Close_lag_7": close_lag_7,
            "Close_roll_mean_7": close_roll_mean_7,
            "Volatility_7": volatility_7,
            "Return_1": return_1,
            "Momentum_3": momentum_3,
            "Momentum_7": momentum_7,
            "Trend_7_14": trend_7_14
        }

        input_features = pd.DataFrame([{col: feature_dict[col] for col in feature_cols}])

        base_pred = model.predict(input_features)[0]

        # =========================
        # IMPORTANT FIX
        # =========================
        trend = last_close_values[-1] - last_close_values[-2]
        noise = np.random.normal(0, max(volatility_7 * 0.25, 0.15))
        return_noise = np.random.normal(avg_return, return_std)

        next_pred = (
            base_pred
            + 0.35 * trend
            + noise
            + last_close_values[-1] * return_noise * 0.25
        )

        # prevent unrealistic jump
        max_change = last_close_values[-1] * 0.04
        lower_limit = last_close_values[-1] - max_change
        upper_limit = last_close_values[-1] + max_change

        next_pred = np.clip(next_pred, lower_limit, upper_limit)

        future_date = last_known_date + pd.Timedelta(days=day)

        if future_date <= today_date:
            forecast_type = "Forecast Until Today"
        else:
            forecast_type = "Future Forecast"

        future_predictions.append({
            "Date": future_date,
            "Predicted_Close": next_pred,
            "Forecast_Type": forecast_type
        })

        last_close_values.append(next_pred)

    return pd.DataFrame(future_predictions)

future_forecast_df = recursive_future_forecast_until_today(
    model=model,
    df_model=df_model,
    feature_cols=feature_cols,
    extra_future_days=extra_future_days
)

# =========================
# UNCERTAINTY RANGE
# =========================
recent_error_std = np.std(y_test.values - y_pred.values)

future_forecast_df["Lower_Bound"] = (
    future_forecast_df["Predicted_Close"] - 1.96 * recent_error_std
)

future_forecast_df["Upper_Bound"] = (
    future_forecast_df["Predicted_Close"] + 1.96 * recent_error_std
)

st.dataframe(future_forecast_df)

# =========================
# FORECAST PLOT
# =========================
fig5, ax5 = plt.subplots(figsize=(14, 6))

ax5.plot(
    df_model["Date"].tail(120),
    df_model["Close"].tail(120),
    label="Historical Close",
    linewidth=3
)

connection_dates = [
    df_model["Date"].iloc[-1],
    future_forecast_df["Date"].iloc[0]
]

connection_values = [
    df_model["Close"].iloc[-1],
    future_forecast_df["Predicted_Close"].iloc[0]
]

ax5.plot(
    connection_dates,
    connection_values,
    linestyle="--",
    linewidth=2,
    label="Forecast Start"
)

until_today_df = future_forecast_df[
    future_forecast_df["Forecast_Type"] == "Forecast Until Today"
]

future_only_df = future_forecast_df[
    future_forecast_df["Forecast_Type"] == "Future Forecast"
]

if len(until_today_df) > 0:
    ax5.plot(
        until_today_df["Date"],
        until_today_df["Predicted_Close"],
        label="Forecast Until Current Date",
        marker="o",
        linewidth=3
    )

if len(future_only_df) > 0:
    ax5.plot(
        future_only_df["Date"],
        future_only_df["Predicted_Close"],
        label="Forecast After Today",
        marker="o",
        linewidth=3
    )

ax5.fill_between(
    future_forecast_df["Date"],
    future_forecast_df["Lower_Bound"],
    future_forecast_df["Upper_Bound"],
    alpha=0.2,
    label="Uncertainty Range"
)

ax5.axvline(
    pd.Timestamp.today().normalize(),
    linestyle="--",
    linewidth=2,
    label="Today"
)

ax5.set_title(f"Forecast Until Current Date and Beyond using {selected_model}")
ax5.set_xlabel("Date")
ax5.set_ylabel("Close Price")
ax5.legend()
ax5.grid(True)

st.pyplot(fig5)

# =========================
# LATEST RECORD
# =========================
st.subheader("Latest Available Record")
st.dataframe(df_model.tail(1))

st.caption("Built with Streamlit for crude oil price forecasting.")
