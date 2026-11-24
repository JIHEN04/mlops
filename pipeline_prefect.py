import argparse
import subprocess
import sys

from prefect import flow, task

import model_pipeline as mp

FILES = ["model_pipeline.py", "main.py", "pipeline_prefect.py"]


def run(cmd):
    """Lance une commande, et fait échouer la task en cas d'erreur."""
    subprocess.run(cmd, check=True)  # nosec B603


# ---------- Tasks : dépendances et qualité du code ----------
@task(name="Installer les dépendances")
def install_dependencies():
    run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


@task(name="Formatage du code")
def format_code():
    run([sys.executable, "-m", "black", *FILES])


@task(name="Qualité du code")
def check_quality():
    run([sys.executable, "-m", "flake8", "--max-line-length=88", *FILES])


@task(name="Sécurité du code")
def check_security():
    run([sys.executable, "-m", "bandit", "-ll", "-q", *FILES])


@task(name="Tests unitaires")
def run_unit_tests():
    run([sys.executable, "-m", "pytest", "tests/", "-v"])


# ---------- Tasks : data et modèle ----------
@task(name="Préparation des données")
def prepare_data(data_path="Churn_Modelling.csv"):
    return mp.prepare_data(data_path)


@task(name="Entraînement du modèle")
def train_model(x_train, y_train):
    return mp.train_model(x_train, y_train)


@task(name="Sauvegarder le modèle")
def save_model(model, path="model.joblib"):
    mp.save_model(model, path)
    return path


@task(name="Charger le modèle")
def load_model(path="model.joblib"):
    return mp.load_model(path)


@task(name="Evaluer le modèle")
def evaluate_model(model, x_test, y_test):
    return mp.evaluate_model(model, x_test, y_test)


# ---------- Flows ----------
@flow(name="install")
def install_flow():
    install_dependencies()


@flow(name="code")
def code_flow():
    install_dependencies()
    format_code()
    check_quality()
    check_security()
    run_unit_tests()


@flow(name="train")
def train_flow():
    x_train, x_test, y_train, y_test = prepare_data()
    model = train_model(x_train, y_train)
    save_model(model)


@flow(name="evaluate")
def evaluate_flow():
    # on a besoin de x_test et y_test pour évaluer
    _, x_test, _, y_test = prepare_data()
    model = load_model()
    evaluate_model(model, x_test, y_test)


@flow(name="all")
def all_flow():
    # étapes liées au code, avant la préparation des données
    install_dependencies()
    format_code()
    check_quality()
    check_security()
    run_unit_tests()
    # étapes data / modèle
    x_train, x_test, y_train, y_test = prepare_data()
    model = train_model(x_train, y_train)
    save_model(model)
    evaluate_model(model, x_test, y_test)


FLOWS = {
    "all": all_flow,
    "code": code_flow,
    "install": install_flow,
    "train": train_flow,
    "entrainement": train_flow,
    "evaluate": evaluate_flow,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline Prefect")
    parser.add_argument("--flow", choices=FLOWS.keys(), required=True)
    args = parser.parse_args()
    FLOWS[args.flow]()
