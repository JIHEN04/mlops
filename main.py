"""Point d'entrée : exécute les étapes du pipeline ML via des arguments CLI."""

import argparse

from model_pipeline import (
    evaluate_model,
    load_model,
    prepare_data,
    save_model,
    train_model,
)


def main():
    parser = argparse.ArgumentParser(description="Pipeline ML - Churn Modelling")
    parser.add_argument(
        "--data", default="Churn_Modelling.csv", help="Chemin du fichier CSV"
    )
    parser.add_argument("--model", default="model.joblib", help="Fichier du modèle")
    parser.add_argument("--prepare", action="store_true", help="Préparer les données")
    parser.add_argument(
        "--train", action="store_true", help="Entraîner et sauvegarder le modèle"
    )
    parser.add_argument(
        "--evaluate", action="store_true", help="Charger et évaluer le modèle"
    )
    parser.add_argument("--all", action="store_true", help="Exécuter toutes les étapes")
    args = parser.parse_args()

    if args.all:
        args.prepare = args.train = args.evaluate = True

    if not (args.prepare or args.train or args.evaluate):
        parser.print_help()
        return

    x_train, x_test, y_train, y_test = prepare_data(args.data)

    if args.prepare:
        print("Données préparées :")
        print(f"  x_train : {x_train.shape}  |  x_test : {x_test.shape}")

    if args.train:
        model = train_model(x_train, y_train)
        print("Modèle entraîné.")
        save_model(model, args.model)

    if args.evaluate:
        model = load_model(args.model)
        evaluate_model(model, x_test, y_test)


if __name__ == "__main__":
    main()
