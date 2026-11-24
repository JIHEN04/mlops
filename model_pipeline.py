"""Pipeline ML modulaire pour la prédiction du churn client (Churn_Modelling.csv)."""

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def prepare_data(data_path="Churn_Modelling.csv", test_size=0.2, random_state=1):
    """Charge le CSV, nettoie et encode les données, puis les sépare en train/test.

    Retourne : x_train, x_test, y_train, y_test
    """
    df = pd.read_csv(data_path)

    # Nettoyage : suppression des doublons et des valeurs manquantes
    df = df.drop_duplicates().dropna()

    # Encodage de la colonne Gender (Female=0, Male=1)
    encoder = LabelEncoder()
    df["Gender"] = encoder.fit_transform(df["Gender"])

    # Suppression des colonnes inutiles pour la prédiction
    df = df.drop(columns=["RowNumber", "CustomerId", "Surname", "Geography"])

    x = df.drop(columns=["Exited"])
    y = df["Exited"]

    return train_test_split(x, y, test_size=test_size, random_state=random_state)


def train_model(x_train, y_train, n_estimators=100, random_state=42):
    """Entraîne un RandomForestClassifier et le retourne."""
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(x_train, y_train)
    return model


def evaluate_model(model, x_test, y_test):
    """Évalue le modèle : affiche l'accuracy, la matrice de confusion et le rapport.

    Retourne l'accuracy.
    """
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy : {accuracy * 100:.2f} %")
    print("Matrice de confusion :")
    print(confusion_matrix(y_test, y_pred))
    print("Rapport de classification :")
    print(classification_report(y_test, y_pred))
    return accuracy


def save_model(model, filename="model.joblib"):
    """Sauvegarde le modèle entraîné avec joblib."""
    joblib.dump(model, filename)
    print(f"Modèle sauvegardé dans {filename}")


def load_model(filename="model.joblib"):
    """Charge un modèle sauvegardé avec joblib."""
    model = joblib.load(filename)
    print(f"Modèle chargé depuis {filename}")
    return model
