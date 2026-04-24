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

    st.error("CSV file not found. Upload crude_oil_final.csv to the repo.")
    st.stop()

df = load_data()

# =========================
# FEATURE SET
# =========================
target_col = "Close"

feature_cols = [
    "Close_lag_1",
    "Close_lag_3",
    "Close_lag_7",
    "Close_roll_mean_7",
    "Volatility_7"
]

feature_cols = [col for col in feature_cols if col in df.columns]

if len(feature_cols) == 0:
    st.error("Required feature columns were not found in the dataset.")
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
# METRICS FUNCTION
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
        n_estimators=300,
        max_depth=10,
        random_state=42
    )
    rf_model.fit(X_train, y_train)

    xgb_model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
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

future_days = st.sidebar.slider(
    "Future forecast days",
    min_value=3,
    max_value=30,
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
# TOP SECTION
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
# HISTORICAL CLOSE CHART
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
# MODEL PERFORMANCE TABLE
# =========================
st.subheader("Model Performance Comparison")
st.dataframe(comparison_df)

# =========================
# CURRENT MODEL METRICS
# =========================
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
ax3.set_title(f"Prediction Error (Actual - Predicted) - {selected_model}")
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
# MANUAL PREDICTION SECTION
# =========================
st.subheader("Manual Prediction")

st.write("Enter feature values to predict the next Close price.")

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

input_array = np.array(input_values).reshape(1, -1)

if st.button("Predict Close Price"):
    prediction = model.predict(input_array)[0]
    st.success(f"Predicted Close Price using {selected_model}: {prediction:.2f}")

# =========================
# FUTURE FORECASTING
# =========================
st.subheader("Future Close Price Forecast")

st.write(
    "This section recursively forecasts future Close prices using the selected model. "
    "Lag features are updated after each predicted day."
)

def recursive_future_forecast(model, df_model, feature_cols, future_days):
    last_known_date = df_model["Date"].iloc[-1]
    last_close_values = list(df_model["Close"].tail(30).values)

    future_predictions = []

    for day in range(1, future_days + 1):
        close_lag_1 = last_close_values[-1]
        close_lag_3 = last_close_values[-3]
        close_lag_7 = last_close_values[-7]
        close_roll_mean_7 = np.mean(last_close_values[-7:])
        volatility_7 = np.std(last_close_values[-7:])

        input_features = pd.DataFrame([{
            "Close_lag_1": close_lag_1,
            "Close_lag_3": close_lag_3,
            "Close_lag_7": close_lag_7,
            "Close_roll_mean_7": close_roll_mean_7,
            "Volatility_7": volatility_7
        }])

        input_features = input_features[feature_cols]

        next_pred = model.predict(input_features)[0]
        future_date = last_known_date + pd.Timedelta(days=day)

        future_predictions.append({
            "Date": future_date,
            "Predicted_Close": next_pred
        })

        last_close_values.append(next_pred)

    return pd.DataFrame(future_predictions)

future_forecast_df = recursive_future_forecast(
    model=model,
    df_model=df_model,
    feature_cols=feature_cols,
    future_days=future_days
)

# uncertainty band using test error standard deviation
recent_error_std = np.std(y_test.values - y_pred.values)

future_forecast_df["Lower_Bound"] = (
    future_forecast_df["Predicted_Close"] - 1.96 * recent_error_std
)

future_forecast_df["Upper_Bound"] = (
    future_forecast_df["Predicted_Close"] + 1.96 * recent_error_std
)

st.dataframe(future_forecast_df)

fig5, ax5 = plt.subplots(figsize=(12, 5))

ax5.plot(
    df_model["Date"].tail(90),
    df_model["Close"].tail(90),
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

ax5.plot(
    future_forecast_df["Date"],
    future_forecast_df["Predicted_Close"],
    label="Future Forecast",
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

ax5.set_title(f"Future Forecast using {selected_model}")
ax5.set_xlabel("Date")
ax5.set_ylabel("Close Price")
ax5.legend()
ax5.grid(True)
st.pyplot(fig5)

# =========================
# LAST RECORD SECTION
# =========================
st.subheader("Latest Available Record")
st.dataframe(df_model.tail(1))

# =========================
# FOOTER
# =========================
st.caption("Built with Streamlit for crude oil price forecasting.")
