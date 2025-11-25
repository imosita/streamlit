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
print(df['target'].value_counts())  # 0 = Maligne, 1 = Bénigne
sns.countplot(x='target', data=df)
plt.title("Répartition Bénin vs Maligne")
plt.show()

# Corrélation
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), cmap='coolwarm', center=0)
plt.title("Matrice de Corrélation")
plt.show()

# Distribution des features clés
for col in ['mean radius', 'mean texture', 'mean perimeter', 'mean area']:
    plt.figure(figsize=(6, 4))
    sns.histplot(data=df, x=col, hue='target', palette='Set1', kde=True)
    plt.title(f"Distribution : {col}")
    plt.show()    
    
    
    
    
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Entraîner un modèle pour obtenir l'importance
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(data.data, data.target)

# Obtenir les 10 meilleures features
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

print("Top 10 features :")
selected_features = []
for i in range(10):
    feature_name = data.feature_names[indices[i]]
    print(f"{i+1}. {feature_name} ({importances[indices[i]]:.3f})")
    selected_features.append(feature_name)

# Sauvegarder les indices
selected_indices = indices[:10]   



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Sélectionner les données
X_selected = data.data[:, selected_indices]
y = data.target

# Split + Scale
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Entraîner
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Sauvegarder
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(selected_indices, 'selected_features.pkl')       