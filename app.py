import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Crude Oil Forecast", layout="wide")

st.title("🛢️ Crude Oil Price Forecasting App")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("crude_oil_final.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    return df

df = load_data()

# =========================
# DISPLAY DATA
# =========================
st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# =========================
# FEATURES
# =========================
feature_cols = [
    "Close_lag_1",
    "Close_lag_3",
    "Close_lag_7",
    "Close_roll_mean_7",
    "Volatility_7"
]

feature_cols = [col for col in feature_cols if col in df.columns]

target = "Close"

df_model = df.dropna()

X = df_model[feature_cols]
y = df_model[target]

# =========================
# MODEL SELECTION
# =========================
st.subheader("⚙️ Select Model")

model_option = st.selectbox(
    "Choose Model",
    ["Random Forest", "XGBoost"]
)

# =========================
# TRAIN MODEL
# =========================
if model_option == "Random Forest":
    model = RandomForestRegressor(n_estimators=200, random_state=42)
else:
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

model.fit(X, y)

st.success(f"{model_option} model trained successfully!")

# =========================
# PREDICTION INPUT
# =========================
st.subheader("🔢 Enter Feature Values")

input_data = []

for col in feature_cols:
    value = st.number_input(
        f"{col}",
        value=float(df_model[col].iloc[-1])
    )
    input_data.append(value)

input_array = np.array(input_data).reshape(1, -1)

# =========================
# PREDICTION BUTTON
# =========================
if st.button("Predict Close Price"):
    prediction = model.predict(input_array)[0]
    st.success(f"💰 Predicted Close Price: {prediction:.2f}")

# =========================
# PLOT HISTORICAL DATA
# =========================
st.subheader("📈 Historical Close Price")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df["Date"], df["Close"], label="Close Price")
ax.set_title("Crude Oil Close Price Over Time")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)