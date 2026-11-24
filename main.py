"""Exécute les étapes du pipeline ML depuis la ligne de commande."""

import argparse
import json
import os
import sys

import model_pipeline as pipeline

MIN_ACCURACY = 0.8


def _log_metrics(
    run_name: str,
    metrics: dict,
    params: dict | None = None,
) -> None:
    """Enregistre une expérience dans le suivi MLflow local."""
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow

    tracking_db = pipeline.ROOT / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{tracking_db}")
    mlflow.set_experiment("iris-classifier")
    with mlflow.start_run(run_name=run_name):
        if params:
            mlflow.log_params(params)
        mlflow.log_metrics(metrics)


def run_prepare() -> None:
    """Prépare les données et affiche la taille des découpages."""
    summary = pipeline.prepare_data()
    print(f"Données prêtes: {summary['n_train']} train / {summary['n_test']} test.")


def run_train() -> None:
    """Entraîne le modèle et le dépose dans le fichier temporaire."""
    x_train, _, y_train, _ = pipeline.load_splits()
    model = pipeline.train_model(x_train, y_train)
    pipeline.save_model(model, pipeline.STAGING_MODEL_PATH)
    accuracy = float(model.score(x_train, y_train))
    _log_metrics(
        "train",
        {"train_accuracy": accuracy},
        {"estimator": "LogisticRegression", "scaler": "StandardScaler"},
    )
    print(f"Modèle entraîné. Accuracy train: {accuracy:.3f}")


def run_save() -> None:
    """Copie le modèle entraîné vers l'emplacement final."""
    model = pipeline.load_model(pipeline.STAGING_MODEL_PATH)
    path = pipeline.save_model(model, pipeline.MODEL_PATH)
    print(f"Modèle sauvegardé: {path}")


def run_load() -> None:
    """Charge le modèle final et affiche ses classes."""
    model = pipeline.load_model(pipeline.MODEL_PATH)
    classes = list(model.named_steps["classifier"].classes_)
    print(f"Modèle chargé. Classes: {', '.join(map(str, classes))}")


def run_evaluate() -> None:
    """Évalue le modèle final et refuse une accuracy trop basse."""
    _, x_test, _, y_test = pipeline.load_splits()
    model = pipeline.load_model(pipeline.MODEL_PATH)
    metrics = pipeline.evaluate_model(model, x_test, y_test)
    _log_metrics("evaluate", metrics)
    pipeline.METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    pipeline.METRICS_PATH.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    print(
        f"Accuracy test: {metrics['accuracy']:.3f} | "
        f"F1 macro: {metrics['f1_macro']:.3f}"
    )
    if metrics["accuracy"] < MIN_ACCURACY:
        print(
            "Qualité du modèle insuffisante: "
            f"{metrics['accuracy']:.3f} < {MIN_ACCURACY}",
            file=sys.stderr,
        )
        sys.exit(1)


def run_predict() -> None:
    """Prédit la classe du premier exemple de test."""
    _, x_test, _, _ = pipeline.load_splits()
    model = pipeline.load_model(pipeline.MODEL_PATH)
    result = pipeline.predict(model, x_test.head(1))
    species = result.loc[0, "species"]
    confidence = float(result.loc[0, "confidence"])
    print(f"Prédiction: {species} (confiance={confidence:.3f})")


def run_all() -> None:
    """Enchaîne préparation, entraînement, sauvegarde, évaluation et prédiction."""
    run_prepare()
    run_train()
    run_save()
    run_load()
    run_evaluate()
    run_predict()


def parse_args() -> argparse.Namespace:
    """Lit les étapes demandées sur la ligne de commande."""
    parser = argparse.ArgumentParser(description="Pipeline ML modulaire.")
    parser.add_argument("--prepare", action="store_true", help="Préparer les données.")
    parser.add_argument("--train", action="store_true", help="Entraîner le modèle.")
    parser.add_argument("--save", action="store_true", help="Sauvegarder le modèle.")
    parser.add_argument("--load", action="store_true", help="Charger le modèle.")
    parser.add_argument("--evaluate", action="store_true", help="Évaluer le modèle.")
    parser.add_argument("--predict", action="store_true", help="Lancer une prédiction.")
    parser.add_argument(
        "--all", action="store_true", help="Exécuter toutes les étapes."
    )
    return parser.parse_args()


def main() -> None:
    """Dispatch les étapes sélectionnées."""
    args = parse_args()
    selected = [
        args.prepare,
        args.train,
        args.save,
        args.load,
        args.evaluate,
        args.predict,
        args.all,
    ]
    if not any(selected):
        raise SystemExit(
            "Choisissez au moins une étape: "
            "--prepare, --train, --save, --load, --evaluate, --predict ou --all."
        )
    if args.all:
        run_all()
        return
    if args.prepare:
        run_prepare()
    if args.train:
        run_train()
    if args.save:
        run_save()
    if args.load:
        run_load()
    if args.evaluate:
        run_evaluate()
    if args.predict:
        run_predict()


if __name__ == "__main__":
    main()
