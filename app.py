# import streamlit as st
# import joblib
# import numpy as np
# from sklearn.datasets import load_breast_cancer

# # Load model, scaler, and feature indices
# model = joblib.load('model.pkl')
# scaler = joblib.load('scaler.pkl')
# data = load_breast_cancer()
# selected_indices = joblib.load('selected_features.pkl')

# # Page configuration
# st.set_page_config(page_title="Live Corp - Breast Cancer Prediction", layout="centered")
# st.markdown("<h1 style='text-align: center; color: #007BFF;'>🩺 Live Corp - Cancer Prediction</h1>", unsafe_allow_html=True)
# st.markdown("<p style='text-align: center;'>AI-powered breast cancer risk assessment</p>", unsafe_allow_html=True)

# # Input: one slider per selected feature
# input_data = []
# st.sidebar.header("🧬 Tumor Characteristics")

# for idx in selected_indices:
#     feature_name = data.feature_names[idx]
#     value = st.sidebar.slider(
#         label=feature_name.replace(" ", " ").title(),
#         min_value=0.0,
#         max_value=50.0,
#         value=15.0,
#         step=0.1
#     )
#     input_data.append(value)

# # Prediction
# if st.button("📊 Analyze Tumor", use_container_width=True):
#     input_array = np.array([input_data])
#     input_scaled = scaler.transform(input_array)
#     pred = model.predict(input_scaled)[0]
#     proba = model.predict_proba(input_scaled)[0]

#     st.markdown("### 📋 Diagnosis Result")
#     if pred == 1:
#         st.success("🟢 **Benign**", icon="✅")
#         st.progress(float(proba[1]), text="Confidence Level")
#     else:
#         st.error("🔴 **Malignant**", icon="🚨")
#         st.progress(float(proba[0]), text="Confidence Level")
#     st.caption(f"AI Confidence: {max(proba):.1%}")

# # Disclaimer
# st.markdown("---")
# st.caption("⚠️ For demonstration only — not a medical device | © Live Corp 2025")   



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
st.set_page_config(page_title="Live Corp - Cancer Prediction", layout="centered")

# Custom CSS for professional styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
        color: #2c3e50;
    }
    h1 {
        color: #007BFF;
        text-align: center;
        font-family: 'Segoe UI', sans-serif;
    }
    .stButton>button {
        background-color: #007BFF;
        color: white;
        border-radius: 8px;
        height: 50px;
        font-size: 18px;
    }
    .stSlider>label {
        font-weight: 600;
        color: #2c3e50;
    }
    .footer {
        text-align: center;
        color: #7f8c8d;
        font-size: 0.9em;
        margin-top: 30px;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1>🩺 Live Corp - Cancer Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #34495e;'>Medical-grade risk assessment system</p>", unsafe_allow_html=True)

# Input section
st.sidebar.title("🔬 Patient Data Input")
st.sidebar.markdown("Adjust tumor characteristics below:")

input_data = []
for idx in selected_indices:
    feature_name = data.feature_names[idx].replace(" mean", "").replace("_", " ").title()
    value = st.sidebar.slider(
        feature_name,
        min_value=0.0,
        max_value=50.0,
        value=15.0,
        step=0.1,
        format="%.1f"
    )
    input_data.append(value)

# Prediction
if st.button("🔍 Run Analysis", use_container_width=True):
    input_array = np.array([input_data])
    input_scaled = scaler.transform(input_array)
    pred = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0]

    st.markdown("## 📊 Prediction Result")
    
    if pred == 1:
        st.markdown(
            """
            <div style='padding: 20px; border-radius: 10px; background-color: #d4edda; color: #155724; text-align: center; font-size: 20px;'>
            ✅ <strong>Benign</strong>
            </div>
            """, 
            unsafe_allow_html=True
        )
        st.progress(float(proba[1]))
    else:
        st.markdown(
            """
            <div style='padding: 20px; border-radius: 10px; background-color: #f8d7da; color: #721c24; text-align: center; font-size: 20px;'>
            ❌ <strong>Malignant</strong>
            </div>
            """, 
            unsafe_allow_html=True
        )
        st.progress(float(proba[0]))
    
    st.markdown(f"<p style='text-align: center; color: #2c3e50;'>Confidence: <strong>{max(proba):.1%}</strong></p>", unsafe_allow_html=True)

# Disclaimer
st.markdown("---")
st.markdown("<p class='footer'>⚠️ For demonstration only — not a medical device | © Live Corp 2025</p>", unsafe_allow_html=True)   