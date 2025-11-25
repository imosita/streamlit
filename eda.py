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

sns.set_theme(style='white')

st.set_page_config(page_title='Live Corp - Explorateur Data & Modèles', page_icon='🧪', layout='wide')

st.markdown("""
<style>
html, body, [class*="css"]  {
    background-color: #f4f8fb !important;
    color: #1d2a44;
}
.sidebar .sidebar-content {
    background-color: #eaf4f6 !important;
}
section.main > div { background: transparent; }
.stMetric label { color: #3a6073; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return data, df

@st.cache_resource
def feature_importance(data):
    rf = RandomForestClassifier(n_estimators=300, random_state=42)
    rf.fit(data.data, data.target)
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    return importances, indices

@st.cache_resource
def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
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

# Data loading
raw_data, df = load_data()
feature_names = raw_data.feature_names

st.title('🧪 Live Corp — Explorateur Données & Modèles')
st.caption("Analyse exploratoire du dataset Cancer du Sein & entraînement de modèles ML pour validation interne")

# Sidebar controls
st.sidebar.header('⚙️ Paramétrage')
feature_count = st.sidebar.slider('Nombre de features importantes', min_value=5, max_value=20, value=10, step=1)
sns_color = st.sidebar.color_picker('Couleur des graphiques', value='#1BA3A1')

st.sidebar.markdown('---')
st.sidebar.markdown('**Mode d\'emploi**')
st.sidebar.markdown("""- Ajustez le nombre de variables
- Explorez les distributions
- Lancez l'entraînement pour exporter le meilleur modèle""")

# Overview section
col1, col2, col3 = st.columns(3)
col1.metric('Observations', df.shape[0])
col2.metric('Variables', df.shape[1] - 1)
col3.metric('Classes', int(df['target'].nunique()))

with st.expander('🗂️ Informations générales', expanded=False):
    st.write(df.head())
    st.write(df.describe())

with st.expander('📊 Répartition des classes', expanded=True):
    class_counts = df['target'].value_counts().rename({0: 'Maligne', 1: 'Bénigne'})
    st.bar_chart(class_counts)

# Feature importance
importances, indices = feature_importance(raw_data)
selected_indices = indices[:feature_count]
selected_features = [feature_names[i] for i in selected_indices]

imp_df = pd.DataFrame({
    'Feature': [feature_names[i] for i in indices],
    'Importance': importances[indices],
})

with st.expander('🏆 Importance des variables', expanded=True):
    st.write(imp_df.head(20))

# Correlation heatmap
with st.expander('🔗 Corrélations', expanded=False):
    corr_matrix = df[selected_features].corr().abs()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(corr_matrix, mask=mask, cmap='BuGn', center=0, ax=ax, linewidths=0.5)
    ax.set_title('Matrice de corrélation (features sélectionnées)')
    st.pyplot(fig)

# Feature distributions
with st.expander('📈 Distributions', expanded=False):
    options = st.multiselect('Choisir les features à visualiser', selected_features, default=selected_features[:3])
    for feat in options:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.kdeplot(data=df, x=feat, hue='target', fill=True, palette=['#ff8a80', sns_color], ax=ax)
        ax.set_title(f'Distribution : {feat}')
        st.pyplot(fig)

st.markdown('---')
st.subheader('🤖 Entraînement & Export des modèles')

if st.button('Lancer l\'entraînement', use_container_width=True):
    X_sel = raw_data.data[:, selected_indices]
    y = raw_data.target
    with st.spinner('Entraînement des modèles...'):
        performances, best_name, best_model, scaler, cm = train_models(X_sel, y)

    perf_df = pd.DataFrame.from_dict(performances, orient='index', columns=['Accuracy']).sort_values('Accuracy', ascending=False)
    st.write('Performances (accuracy) :', perf_df)
    st.success(f'Meilleur modèle : {best_name} ({performances[best_name]:.2%})')

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', ax=ax,
                xticklabels=['Maligne', 'Bénigne'], yticklabels=['Maligne', 'Bénigne'])
    ax.set_ylabel('Vraie classe')
    ax.set_xlabel('Classe prédite')
    ax.set_title(f'Matrice de confusion — {best_name}')
    st.pyplot(fig)

    # Save artefacts to disk and provide downloads
    joblib.dump(best_model, 'model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(selected_indices, 'selected_features.pkl')

    st.markdown('Artefacts enregistrés (model.pkl, scaler.pkl, selected_features.pkl).')

    # In-memory downloads for convenience
    model_buffer = io.BytesIO()
    scaler_buffer = io.BytesIO()
    joblib.dump(best_model, model_buffer)
    joblib.dump(scaler, scaler_buffer)

    st.download_button('Télécharger le modèle', data=model_buffer.getvalue(), file_name='model.pkl')
    st.download_button('Télécharger le scaler', data=scaler_buffer.getvalue(), file_name='scaler.pkl')
    st.download_button('Télécharger les indices', data=np.array(selected_indices).tobytes(), file_name='selected_features.npy')
else:
    st.info('Définissez vos paramètres puis lancez l\'entraînement pour produire les artefacts.')

st.markdown('---')
st.markdown("<p style='text-align:center; color:#6b7a84;'>© Live Corp 2025 — Analyse interne</p>", unsafe_allow_html=True)