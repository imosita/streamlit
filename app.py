import streamlit as st
import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer

# Charger les composants
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
data = load_breast_cancer()
selected_indices = joblib.load('selected_features.pkl')

# Configuration
st.set_page_config(page_title="Live Corp - Diagnostic IA", layout="centered")
st.markdown("<h1 style='text-align: center; color: #007BFF;'>🩺 Diagnostic IA - Cancer du Sein</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Analyse assistée par intelligence artificielle</p>", unsafe_allow_html=True)

# Sélection des features
st.sidebar.header("🔧 Paramètres du Patient")
input_data = []

# Organiser les sliders par groupe
st.sidebar.subheader("📏 Dimensions de la tumeur")
radius = st.sidebar.slider("Rayon moyen", 0.0, 30.0, 14.0)
perimeter = st.sidebar.slider("Périmètre moyen", 0.0, 200.0, 90.0)
area = st.sidebar.slider("Aire moyenne", 0.0, 2500.0, 700.0)

st.sidebar.subheader("📊 Texture & Forme")
texture = st.sidebar.slider("Texture moyenne", 0.0, 40.0, 18.0)
concavity = st.sidebar.slider("Concavité moyenne", 0.0, 5.0, 0.3)
concave_points = st.sidebar.slider("Points concaves moyens", 0.0, 1.0, 0.1)

input_data = [radius, texture, perimeter, area, concavity, concave_points] + [15.0] * 4  # Compléter pour 10

# Prédiction
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🔍 Lancer l'analyse", key="predict", use_container_width=True):
        input_array = np.array([input_data])
        input_scaled = scaler.transform(input_array)
        pred = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0]

        if pred == 1:
            st.success("✅ **Résultat : Bénin**", icon="🟢")
            st.progress(float(proba[1]), text="Confiance du diagnostic")
        else:
            st.error("⚠️ **Résultat : Maligne**", icon="🔴")
            st.progress(float(proba[0]), text="Confiance du diagnostic")
        st.caption(f"Probabilité : {max(proba):.1%}")

# Info éthique
st.markdown("---")
st.caption("ℹ️ Ce modèle est un outil d'aide au diagnostic. Ne remplace pas un avis médical. | © Live Corp 2025")   