
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model and scaler
model = joblib.load("water_model.pkl")
scaler = joblib.load("scaler.pkl")

# Page Configuration
st.set_page_config(
    page_title="Water Quality Prediction",
    page_icon="💧",
    layout="centered"
)

# Sidebar
st.sidebar.title("About Project")

st.sidebar.info("""
💧 Water Quality Prediction System

Algorithm Used:
✔ XGBoost
✔ SMOTE
✔ StandardScaler

Accuracy: 70.6%

Developed using Streamlit
""")

st.sidebar.markdown("---")
st.sidebar.subheader("Developer")
st.sidebar.write("Arpita Mahajan")
st.sidebar.write("AI & Machine Learning Researcher")

# Main Title
st.title("💧 Water Quality Prediction System")

st.success("✅ Model Accuracy: 70.6%")

st.write("Enter the water parameters below:")

# CSV Upload
uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

# Manual Input
ph = st.number_input("pH", value=7.0)
hardness = st.number_input("Hardness", value=200.0)
solids = st.number_input("Solids", value=20000.0)
chloramines = st.number_input("Chloramines", value=7.0)
sulfate = st.number_input("Sulfate", value=350.0)
conductivity = st.number_input("Conductivity", value=500.0)
organic_carbon = st.number_input("Organic Carbon", value=15.0)
trihalomethanes = st.number_input("Trihalomethanes", value=70.0)
turbidity = st.number_input("Turbidity", value=4.0)

# Single Prediction
if st.button("Predict"):

    sample = np.array([[ph,
                        hardness,
                        solids,
                        chloramines,
                        sulfate,
                        conductivity,
                        organic_carbon,
                        trihalomethanes,
                        turbidity]])

    sample_scaled = scaler.transform(sample)

    prediction = model.predict(sample_scaled)

    probability = model.predict_proba(sample_scaled)

    confidence = np.max(probability) * 100

    st.info(f"Prediction Confidence: {confidence:.2f}%")

    st.write(f"💧 Probability of Drinkable Water: {probability[0][1]*100:.2f}%")

    st.write(f"❌ Probability of Non-Drinkable Water: {probability[0][0]*100:.2f}%")

    if prediction[0] == 1:
        st.balloons()
        st.success("✅ Water is Safe for Drinking")
    else:
        st.error("❌ Water is Not Safe for Drinking")

# CSV Batch Prediction
if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")
    st.write(data.head())

    # Remove target column if present
    if "Potability" in data.columns:
        data = data.drop("Potability", axis=1)

    # Correct column order
    expected_columns = [
        'ph',
        'Hardness',
        'Solids',
        'Chloramines',
        'Sulfate',
        'Conductivity',
        'Organic_carbon',
        'Trihalomethanes',
        'Turbidity'
    ]

    data = data[expected_columns]

    data_scaled = scaler.transform(data)

    predictions = model.predict(data_scaled)

    data["Prediction"] = predictions

    st.subheader("Prediction Results")
    st.write(data)

    csv = data.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Prediction Results",
        data=csv,
        file_name="prediction_results.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.write("Developed using Machine Learning and Streamlit")
