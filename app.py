import io
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Caching assets for faster reloads
@st.cache_resource
def load_assets():
    data = load_breast_cancer()
    try:
        model = joblib.load("model.pkl")
        scaler = joblib.load("scaler.pkl")
        selected = joblib.load("selected_features.pkl")
        selected = list(np.array(selected, dtype=int).flatten())
        return model, scaler, selected, data, None
    except Exception as err:
        return None, None, None, data, str(err)

@st.cache_data
def load_data():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    return data, df

@st.cache_resource
def feature_importance(dataset):
    rf = RandomForestClassifier(n_estimators=300, random_state=42)
    rf.fit(dataset.data, dataset.target)
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    return importances, indices

@st.cache_resource
def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Régression Logistique": LogisticRegression(max_iter=1000),
        "Forêt Aléatoire": RandomForestClassifier(n_estimators=300, random_state=42),
        "SVM": SVC(probability=True),
        "KNN": KNeighborsClassifier(),
        "Réseau de Neurones": MLPClassifier(max_iter=1000, random_state=42),
    }

    performances = {}
    trained = {}
    for name, mdl in models.items():
        mdl.fit(X_train_scaled, y_train)
        preds = mdl.predict(X_test_scaled)
        performances[name] = accuracy_score(y_test, preds)
        trained[name] = mdl

    best_name = max(performances, key=performances.get)
    best_model = trained[best_name]
    cm = confusion_matrix(y_test, best_model.predict(X_test_scaled))

    return performances, best_name, best_model, scaler, cm

model, scaler, selected_indices, asset_data, load_error = load_assets()
raw_data, df = load_data()
feature_names = raw_data.feature_names

st.set_page_config(page_title="Live Corp - Diagnostic & Exploration", page_icon="🩺", layout="wide")

# Simple professional CSS
st.markdown(
    """
    <style>
    .main { background-color: #f7fbfc; color: #243746; font-family: 'Inter', sans-serif; }
    h1 { color: #0b3d91; text-align: center; }
    .stButton>button { background-color: #0b3d91; color: #fff; border-radius: 8px; border: none; }
    .stButton>button:hover { background-color: #0a3377; }
    .card { padding: 16px; border-radius: 12px; background: #ffffff; box-shadow: 0 1px 3px rgba(16,24,40,0.08); }
    </style>
    """,
    unsafe_allow_html=True,
)

sns.set_theme(style="whitegrid")

st.sidebar.title("Live Corp")
view = st.sidebar.radio("Navigation", ["Diagnostic IA", "Exploration Data"])

st.sidebar.markdown("---")

if view == "Diagnostic IA":
    st.sidebar.subheader("État des artefacts")
    if load_error:
        st.sidebar.warning("Modèle absent. Générez-le via l'onglet Exploration.")
    else:
        st.sidebar.success("Modèle et scaler chargés.")

    st.markdown("<h1>🩺 Diagnostic IA — Cancer du Sein</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center; color:#3d5568'>Évaluation assistée par IA pour la détection précoce (usage démonstration seulement)</p>",
        unsafe_allow_html=True,
    )

    if load_error:
        st.error(f"Impossible de charger les artefacts ML : {load_error}")
        st.info("Rendez-vous dans l'onglet Exploration pour entraîner et exporter un nouveau modèle.")
    else:
        def humanize(name: str) -> str:
            return name.replace("_", " ").replace(" mean", "").title()

        feature_info = []
        for idx in selected_indices:
            values = raw_data.data[:, idx]
            feature_info.append(
                {
                    "idx": idx,
                    "label": humanize(feature_names[idx]),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "median": float(np.median(values)),
                }
            )

        left_col, right_col = st.columns((2, 1))

        with left_col:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("🔬 Paramètres du patient")
            with st.form(key="diagnostic_form"):
                inputs = {}
                for info in feature_info:
                    span = info["max"] - info["min"]
                    step = float(round(max(span / 200, 0.001), 6))
                    inputs[info["idx"]] = st.slider(
                        label=info["label"],
                        min_value=float(info["min"]),
                        max_value=float(info["max"]),
                        value=float(np.clip(info["median"], info["min"], info["max"])),
                        step=step,
                    )
                submitted = st.form_submit_button("🔍 Lancer l'analyse")
            st.markdown("</div>", unsafe_allow_html=True)

            if submitted:
                ordered_vals = [inputs[info["idx"]] for info in feature_info]
                input_array = np.array([ordered_vals])
                input_scaled = scaler.transform(input_array)
                pred = int(model.predict(input_scaled)[0])
                proba = model.predict_proba(input_scaled)[0]

                st.markdown("<div class='card' style='margin-top:16px;'>", unsafe_allow_html=True)
                st.subheader("📊 Résultat du diagnostic")
                label = "Bénin" if pred == 1 else "Maligne"
                color = "green" if pred == 1 else "red"
                st.markdown(f"<p style='font-size:20px; color:{color};'><strong>{label}</strong></p>", unsafe_allow_html=True)

                prob_benign = float(proba[1])
                prob_malign = float(proba[0])
                st.write("Confiance — Bénin :", f"{prob_benign:.1%}")
                st.progress(prob_benign)
                st.write("Confiance — Maligne :", f"{prob_malign:.1%}")
                st.progress(prob_malign)

                df_inputs = pd.DataFrame(
                    [ordered_vals],
                    columns=[humanize(feature_names[info["idx"]]) for info in feature_info],
                )
                st.download_button(
                    "Télécharger les paramètres (CSV)",
                    data=df_inputs.to_csv(index=False).encode("utf-8"),
                    file_name="input_parameters.csv",
                )
                st.markdown("</div>", unsafe_allow_html=True)

        with right_col:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("📈 Vue investisseur")
            st.markdown(f"**Modèle** : {model.__class__.__name__}")
            st.markdown(f"**Features utilisées** : {len(selected_indices)}")
            st.markdown(f"**Jeu d'entraînement original** : {raw_data.data.shape[0]} échantillons")

            summary_df = pd.DataFrame(
                raw_data.data[:, selected_indices],
                columns=[humanize(feature_names[idx]) for idx in selected_indices],
            )
            st.markdown("**Statistiques principales (features retenues)**")
            st.dataframe(summary_df.describe().T[["min", "25%", "50%", "75%", "max"]])
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        "<p style='text-align:center; color:#6b7a84;'>⚠️ Outil de démonstration uniquement — ne remplace pas un avis médical | © Live Corp 2025</p>",
        unsafe_allow_html=True,
    )

else:
    st.sidebar.subheader("Paramétrage EDA")
    feature_count = st.sidebar.slider("Nombre de features importantes", min_value=5, max_value=20, value=10, step=1)
    sns_color = st.sidebar.color_picker("Couleur des graphiques", value="#0b3d91")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Mode d'emploi**")
    st.sidebar.markdown(
        """- Ajustez le nombre de variables sélectionnées
- Explorez les graphiques et statistiques
- Lancez l'entraînement pour générer de nouveaux artefacts"""
    )

    st.title("🧪 Live Corp — Explorateur Données & Modèles")
    st.caption("Analyse exploratoire du dataset Cancer du Sein & validation interne des modèles ML")

    col1, col2, col3 = st.columns(3)
    col1.metric("Observations", df.shape[0])
    col2.metric("Variables", df.shape[1] - 1)
    col3.metric("Classes", int(df["target"].nunique()))

    with st.expander("🗂️ Informations générales", expanded=False):
        st.write(df.head())
        st.write(df.describe())

    with st.expander("📊 Répartition des classes", expanded=True):
        class_counts = df["target"].value_counts().rename({0: "Maligne", 1: "Bénigne"})
        st.bar_chart(class_counts)

    importances, indices = feature_importance(raw_data)
    selected_indices = indices[:feature_count]
    selected_features = [feature_names[i] for i in selected_indices]

    imp_df = pd.DataFrame(
        {
            "Feature": [feature_names[i] for i in indices],
            "Importance": importances[indices],
        }
    )

    with st.expander("🏆 Importance des variables", expanded=True):
        st.write(imp_df.head(20))

    with st.expander("🔗 Corrélations", expanded=False):
        corr_matrix = df[selected_features].corr().abs()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(corr_matrix, mask=mask, cmap="coolwarm", center=0, ax=ax, linewidths=0.5)
        ax.set_title("Matrice de corrélation (features sélectionnées)")
        st.pyplot(fig)

    with st.expander("📈 Distributions", expanded=False):
        options = st.multiselect(
            "Choisir les features à visualiser",
            options=selected_features,
            default=selected_features[: min(3, len(selected_features))],
        )
        if options:
            for feat in options:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.kdeplot(data=df, x=feat, hue="target", fill=True, palette=["#c0392b", sns_color], ax=ax)
                ax.set_title(f"Distribution : {feat}")
                st.pyplot(fig)
        else:
            st.info("Sélectionnez au moins une feature pour afficher la distribution.")

    st.markdown("---")
    st.subheader("🤖 Entraînement & Export des modèles")

    if st.button("Lancer l'entraînement", use_container_width=True):
        X_sel = raw_data.data[:, selected_indices]
        y = raw_data.target
        with st.spinner("Entraînement des modèles..."):
            performances, best_name, best_model, scaler, cm = train_models(X_sel, y)

        perf_df = (
            pd.DataFrame.from_dict(performances, orient="index", columns=["Accuracy"])
            .sort_values("Accuracy", ascending=False)
        )
        st.write("Performances (accuracy) :", perf_df)
        st.success(f"Meilleur modèle : {best_name} ({performances[best_name]:.2%})")

        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            xticklabels=["Maligne", "Bénigne"],
            yticklabels=["Maligne", "Bénigne"],
        )
        ax.set_ylabel("Vraie classe")
        ax.set_xlabel("Classe prédite")
        ax.set_title(f"Matrice de confusion — {best_name}")
        st.pyplot(fig)

        joblib.dump(best_model, "model.pkl")
        joblib.dump(scaler, "scaler.pkl")
        joblib.dump(selected_indices, "selected_features.pkl")

        st.markdown("Artefacts enregistrés (model.pkl, scaler.pkl, selected_features.pkl).")

        model_buffer = io.BytesIO()
        scaler_buffer = io.BytesIO()
        indices_buffer = io.BytesIO()

        joblib.dump(best_model, model_buffer)
        joblib.dump(scaler, scaler_buffer)
        joblib.dump(selected_indices, indices_buffer)

        model_buffer.seek(0)
        scaler_buffer.seek(0)
        indices_buffer.seek(0)

        st.download_button("Télécharger le modèle", data=model_buffer.getvalue(), file_name="model.pkl")
        st.download_button("Télécharger le scaler", data=scaler_buffer.getvalue(), file_name="scaler.pkl")
        st.download_button("Télécharger les indices", data=indices_buffer.getvalue(), file_name="selected_features.pkl")
    else:
        st.info("Définissez vos paramètres puis lancez l'entraînement pour produire les artefacts.")

    st.markdown("---")
    st.markdown(
        "<p style='text-align:center; color:#6b7a84;'>© Live Corp 2025 — Analyse interne</p>",
        unsafe_allow_html=True,
    )