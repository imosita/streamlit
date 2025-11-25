from sklearn.datasets import load_breast_cancer
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Charger les données
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Info générale
print("Informations :")
print(df.info())
print("\nStatistiques :")
print(df.describe())

# Répartition des classes
print("\nRépartition :")
print(df['target'].value_counts())
sns.countplot(x='target', data=df)
plt.title("Répartition Bénin vs Maligne")
plt.show()

# Corrélation
# plt.figure(figsize=(12, 10))
# sns.heatmap(df.corr(), cmap='coolwarm', center=0)
# plt.title("Matrice de Corrélation")
# plt.show()

# Compute correlation matrix
corr_matrix = df.corr().abs()
import numpy as np

# Mask upper triangle
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', center=0)
plt.title("Feature Correlation Matrix")
plt.show()

# Distribution des features clés
for col in ['mean radius', 'mean texture', 'mean perimeter', 'mean area']:
    plt.figure(figsize=(6, 4))
    sns.histplot(data=df, x=col, hue='target', palette='Set1', kde=True)
    plt.title(f"Distribution : {col}")
    plt.show()

# Sélection des meilleures features
from sklearn.ensemble import RandomForestClassifier
import numpy as np

model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(data.data, data.target)

importances = model_rf.feature_importances_
indices = np.argsort(importances)[::-1]

print("Top 10 features :")
for i in range(10):
    print(f"{i+1}. {data.feature_names[indices[i]]} ({importances[indices[i]]:.3f})")

selected_indices = indices[:10]

# Préparation des données avec les features sélectionnées
X_selected = data.data[:, selected_indices]
y = data.target

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entraînement et comparaison des modèles
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

models = {
    "Régression Logistique": LogisticRegression(),
    "Forêt Aléatoire": RandomForestClassifier(),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier(),
    "Réseau de Neurones": MLPClassifier(max_iter=500)
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    results[name] = acc
    print(f"{name}: {acc:.3f}")

# Meilleur modèle
best_model_name = max(results, key=results.get)
print(f"Meilleur modèle : {best_model_name}")

# Matrice de confusion
best_model = models[best_model_name]
y_pred = best_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Maligne', 'Bénigne'],
            yticklabels=['Maligne', 'Bénigne'])
plt.title(f'Matrice de Confusion - {best_model_name}')
plt.ylabel('Vraie Classe')
plt.xlabel('Classe Prédite')
plt.show()

# Sauvegarde du meilleur modèle et du scaler
import joblib
joblib.dump(best_model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(selected_indices, 'selected_features.pkl')   