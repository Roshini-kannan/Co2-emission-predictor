
import streamlit as st
import joblib
import pandas as pd

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Solar Power Prediction",
    page_icon="☀️",
    layout="centered"
)

# -----------------------------
# Load the trained ML model
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("solar_power_prediction_model.pkl")

model = load_model()

# -----------------------------
# Application Title
# -----------------------------
st.title("☀️ Solar Power Prediction")
st.write(
    "Enter the environmental conditions below "
    "to predict the solar power output."
)

st.divider()

# -----------------------------
# User Inputs
# -----------------------------
temperature = st.number_input(
    "🌡️ Temperature",
    min_value=-50.0,
    max_value=100.0,
    value=25.0,
    step=0.1
)

humidity = st.number_input(
    "💧 Humidity",
    min_value=0.0,
    max_value=100.0,
    value=50.0,
    step=0.1
)

solar_irradiance = st.number_input(
    "☀️ Solar Irradiance",
    min_value=0.0,
    value=500.0,
    step=1.0
)

wind_speed = st.number_input(
    "💨 Wind Speed",
    min_value=0.0,
    value=5.0,
    step=0.1
)

st.divider()

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔮 Predict Solar Power", type="primary"):

    input_data = pd.DataFrame(
        [[
            temperature,
            humidity,
            solar_irradiance,
            wind_speed
        ]],
        columns=[
            "temperature",
            "humidity",
            "solar_irradiance",
            "wind_speed"
        ]
    )

    try:
        prediction = model.predict(input_data)[0]

        # Prevent negative prediction
        prediction = max(0, prediction)

        st.success(
            f"☀️ Predicted Solar Power Output: {prediction:.2f}"
        )

    except Exception as e:
        st.error(f"Prediction error: {e}")
