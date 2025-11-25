import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

# Caching assets for faster reloads
@st.cache_resource
def load_assets():
    data = load_breast_cancer()
    try:
        model = joblib.load('model.pkl')
        scaler = joblib.load('scaler.pkl')
        selected = joblib.load('selected_features.pkl')
    except Exception as e:
        return None, None, None, data, str(e)
    return model, scaler, selected, data, None

model, scaler, selected_indices, data, load_error = load_assets()

st.set_page_config(page_title="Live Corp - Diagnostic Cancer", page_icon="🩺", layout="wide")

# Simple professional CSS
st.markdown(
    """
    <style>
    .main { background-color: #f7fbfc; color: #243746; font-family: 'Inter', sans-serif; }
    h1 { color: #0b3d91; text-align: center; }
    .stButton>button { background-color: #0b3d91; color: #fff; border-radius: 8px; }
    .metric-label { color: #5b6b73; }
    .card { padding: 12px; border-radius: 10px; background: #ffffff; box-shadow: 0 1px 3px rgba(16,24,40,0.05); }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown("<h1>🩺 Diagnostic IA — Cancer du Sein</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#3d5568'>Application de démonstration pour évaluation clinique (non médicale)</p>", unsafe_allow_html=True)

if load_error:
    st.error(f"Impossible de charger les artefacts ML: {load_error}. Assurez-vous que model.pkl, scaler.pkl et selected_features.pkl sont présents.")
    st.stop()

# Helpers
def humanize(name: str) -> str:
    return name.replace("_", " ").replace(" mean", "").title()

# Prepare feature bounds from the dataset for nicer sliders
X = data.data
feature_names = data.feature_names

selected_indices = list(selected_indices)
feature_info = []
for idx in selected_indices:
    arr = X[:, idx]
    lo, hi = float(np.min(arr)), float(np.max(arr))
    default = float(np.median(arr))
    feature_info.append({"idx": idx, "name": humanize(feature_names[idx]), "min": lo, "max": hi, "default": default})

# Layout: two columns
left, right = st.columns((2, 1))

with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🔬 Paramètres du patient")

    # Use a form so investor can change many sliders before submit
    with st.form(key='input_form'):
        inputs = {}
        for f in feature_info:
            inputs[f['name']] = st.slider(
                label=f['name'],
                min_value=round(f['min'], 3),
                max_value=round(f['max'], 3),
                value=round(f['default'], 3),
                step=round((f['max'] - f['min']) / 200, 6) if f['max'] > f['min'] else 0.1,
            )

        submitted = st.form_submit_button('🔍 Lancer l\'analyse')
    st.markdown("</div>", unsafe_allow_html=True)

    if submitted:
        # Build input array in the order of selected_indices
        input_vals = [inputs[humanize(feature_names[idx])] for idx in selected_indices]
        input_array = np.array([input_vals])
        input_scaled = scaler.transform(input_array)
        pred = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0]

        # Result card
        st.markdown("<div class='card' style='margin-top:12px;'>", unsafe_allow_html=True)
        st.subheader("📊 Résultat")
        label = 'Bénin' if pred == 1 else 'Maligne'
        color = 'green' if pred == 1 else 'red'
        st.markdown(f"<p style='font-size:20px; color:{color};'><strong>{label}</strong></p>", unsafe_allow_html=True)

        # Show probabilities with clear labels
        prob_benign = float(proba[1])
        prob_malign = float(proba[0])
        st.write("Confiance — Bénin:", f"{prob_benign:.1%}")
        st.progress(prob_benign)
        st.write("Confiance — Maligne:", f"{prob_malign:.1%}")
        st.progress(prob_malign)

        # Download inputs for record / investor
        df_in = pd.DataFrame([input_vals], columns=[humanize(feature_names[idx]) for idx in selected_indices])
        st.download_button("Télécharger les paramètres (CSV)", df_in.to_csv(index=False).encode('utf-8'), file_name='input_parameters.csv')
        st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📈 Vue Investisseur")
    # Model metadata
    st.markdown(f"**Modèle**: {model.__class__.__name__}")
    try:
        n_features = len(selected_indices)
        st.markdown(f"**Features utilisées**: {n_features}")
        st.markdown(f"**Données d\'entrainement**: {X.shape[0]} échantillons, {X.shape[1]} features")
    except Exception:
        pass

    # Quick dataset summary for the selected features
    sample_table = pd.DataFrame(X[:, selected_indices], columns=[humanize(feature_names[idx]) for idx in selected_indices])
    st.markdown("**Aperçu des distributions (quelques statistiques)**")
    st.dataframe(sample_table.describe().T[['min', '25%', '50%', '75%', 'max']])

    st.markdown("</div>", unsafe_allow_html=True)

# Footer / disclaimer
st.markdown("---")
st.markdown("<p style='text-align:center; color:#6b7a84;'>⚠️ Outil de démonstration uniquement — ne remplace pas un avis médical | © Live Corp 2025</p>", unsafe_allow_html=True)