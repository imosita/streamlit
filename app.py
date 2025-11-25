import streamlit as st
import joblib
import numpy as np

# Charger modèle et scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Interface
st.set_page_config(page_title="Live Corp - Cancer Prediction", layout="wide")
st.title("🔬 Prédiction du Cancer du Sein - Démo Investisseur")

# Saisie utilisateur
st.sidebar.header("Entrez les caractéristiques")
input_data = []
for feature in [
    "Rayon moyen", "Texture moyenne", "Périmètre moyen",
    "Aire moyenne", "Lissité moyenne", "Compacité moyenne",
    "Concavité moyenne", "Points concaves moyens", "Symétrie moyenne", "Dimension fractale moyenne"
]:
    value = st.sidebar.slider(feature, 0.0, 50.0, 15.0)
    input_data.append(value)

# Prédiction
if st.button("🔍 Prédire le Diagnostic"):
    input_array = np.array([input_data])
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    # Afficher résultat
    if prediction == 1:
        st.success("✅ **Diagnostic : Bénin**")
        st.progress(float(probability[1]))
    else:
        st.error("⚠️ **Diagnostic : Maligne**")
        st.progress(float(probability[0]))
    st.caption(f"Confiance : {max(probability):.1%}")   