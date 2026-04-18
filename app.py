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
st.write("Compare models, inspect data, and predict crude oil close price.")

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
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
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
    plt.xticks(rotation=45)
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
# LAST RECORD SECTION
# =========================
st.subheader("Latest Available Record")
st.dataframe(df_model.tail(1))

# =========================
# FOOTER
# =========================
st.caption("Built with Streamlit for crude oil price forecasting.")
