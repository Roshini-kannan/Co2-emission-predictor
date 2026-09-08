
import streamlit as st
import joblib
import numpy as np

# ------------------------------------------------------------
# Page Configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="Sustainability Prediction",
    page_icon="🌱",
    layout="centered"
)

# ------------------------------------------------------------
# Model Path
# ------------------------------------------------------------
MODEL_PATH = "polynomialRegModel.pkl"

# ------------------------------------------------------------
# Load Model
# ------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except Exception as e:
    st.error("Could not load the trained model.")
    st.code(str(e))
    st.stop()


# ------------------------------------------------------------
# Application
# ------------------------------------------------------------
st.title("🌱 Sustainability Prediction System")

st.write(
    "This application uses a trained Polynomial Regression "
    "machine learning model to make sustainability-related predictions."
)

st.divider()


# ------------------------------------------------------------
# Input
# ------------------------------------------------------------
st.subheader("Enter Input Details")

input_value = st.number_input(
    "Enter the input value",
    value=0.0,
    step=0.1
)


# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------
if st.button("Predict", type="primary"):

    try:
        # Convert input into NumPy array
        input_data = np.array([[input_value]])

        # Generate prediction
        prediction = model.predict(input_data)

        result = float(prediction[0])

        st.success("Prediction completed successfully!")

        st.metric(
            label="Predicted Sustainability Value",
            value=f"{result:.4f}"
        )

    except Exception as e:
        st.error("Prediction could not be completed.")
        st.code(str(e))


# ------------------------------------------------------------
# Information
# ------------------------------------------------------------
st.divider()

st.subheader("About the Project")

st.write(
    """
    This sustainability project uses Machine Learning to predict
    a sustainability-related value from user-provided input.

    Model:
    Polynomial Regression

    Framework:
    Streamlit

    Programming Language:
    Python
    """
)

st.caption("Sustainability Prediction Project")
