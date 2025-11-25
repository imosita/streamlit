from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Charger les données
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Prétraiter
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Entraîner
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train_scaled, y_train)

# Sauvegarder
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')   