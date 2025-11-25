import streamlit as st
import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer

# Load model, scaler, and feature indices
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
data = load_breast_cancer()
selected_indices = joblib.load('selected_features.pkl')

# Page configuration
st.set_page_config(page_title="Live Corp - Breast Cancer Prediction", layout="centered")
st.markdown("<h1 style='text-align: center; color: #007BFF;'>🩺 Live Corp - Cancer Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>AI-powered breast cancer risk assessment</p>", unsafe_allow_html=True)

# Input: one slider per selected feature
input_data = []
st.sidebar.header("🧬 Tumor Characteristics")

for idx in selected_indices:
    feature_name = data.feature_names[idx]
    value = st.sidebar.slider(
        label=feature_name.replace(" ", " ").title(),
        min_value=0.0,
        max_value=50.0,
        value=15.0,
        step=0.1
    )
    input_data.append(value)

# Prediction
if st.button("📊 Analyze Tumor", use_container_width=True):
    input_array = np.array([input_data])
    input_scaled = scaler.transform(input_array)
    pred = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0]

    st.markdown("### 📋 Diagnosis Result")
    if pred == 1:
        st.success("🟢 **Benign**", icon="✅")
        st.progress(float(proba[1]), text="Confidence Level")
    else:
        st.error("🔴 **Malignant**", icon="🚨")
        st.progress(float(proba[0]), text="Confidence Level")
    st.caption(f"AI Confidence: {max(proba):.1%}")

# Disclaimer
st.markdown("---")
st.caption("⚠️ For demonstration only — not a medical device | © Live Corp 2025")   