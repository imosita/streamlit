import argparse
import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def select_features(data, top_k):
    rf = RandomForestClassifier(n_estimators=300, random_state=42)
    rf.fit(data.data, data.target)
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1][:top_k]
    return indices.tolist(), importances[indices].tolist()


def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=42),
        "svm": SVC(probability=True),
        "knn": KNeighborsClassifier(),
        "mlp": MLPClassifier(max_iter=1000, random_state=42),
    }

    best_name = None
    best_score = -1.0
    best_model = None
    scores = {}

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        score = accuracy_score(y_test, preds)
        scores[name] = score
        if score > best_score:
            best_score = score
            best_name = name
            best_model = model

    return best_name, best_model, scaler, scores


def main(top_features):
    data = load_breast_cancer()
    selected_indices, _ = select_features(data, top_features)

    X_selected = data.data[:, selected_indices]
    y = data.target

    best_name, best_model, scaler, scores = train_models(X_selected, y)

    joblib.dump(best_model, "model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(selected_indices, "selected_features.pkl")

    print("Artefacts générés : model.pkl, scaler.pkl, selected_features.pkl")
    print(f"Meilleur modèle : {best_name} (accuracy={scores[best_name]:.3f})")
    print("Scores détaillés :")
    for name, score in scores.items():
        print(f" - {name}: {score:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Génère les artefacts ML pour l'application Streamlit.")
    parser.add_argument("--top-features", type=int, default=10, help="Nombre de features à conserver")
    args = parser.parse_args()

    main(args.top_features)