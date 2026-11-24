"""Orchestre le pipeline MLOps avec Prefect."""

import argparse
import subprocess
import sys
from pathlib import Path

from prefect import flow, get_run_logger, task

ROOT = Path(__file__).resolve().parent
PYTHON_BIN = Path(sys.executable)
MAIN_PY = ROOT / "main.py"
PROJECT_FILES = [
    ROOT / "model_pipeline.py",
    ROOT / "main.py",
    ROOT / "pipeline_prefect.py",
]


def _run(command: list[str]) -> None:
    """Exécute une commande dans le dossier du projet."""
    logger = get_run_logger()
    logger.info("Commande: %s", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


@task(name="fetch-project")
def fetch_project() -> None:
    """Récupère le projet depuis le dépôt git distant, s'il est configuré."""
    logger = get_run_logger()
    if not (ROOT / ".git").exists():
        logger.info("Aucun dépôt git local. Récupération ignorée.")
        return

    remote = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if remote.returncode != 0:
        logger.info("Aucun remote origin. Récupération ignorée.")
        return

    logger.info("Récupération depuis %s", remote.stdout.strip())
    subprocess.run(["git", "fetch", "origin"], cwd=ROOT, check=True)
    pull = subprocess.run(
        ["git", "pull", "--ff-only"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if pull.returncode != 0:
        logger.warning("git pull non appliqué: %s", pull.stderr.strip())
        return
    logger.info(pull.stdout.strip() or "Dépôt déjà à jour.")


@task(name="install-dependencies")
def install_dependencies() -> None:
    """Installe les dépendances listées dans requirements.txt."""
    logger = get_run_logger()
    _run([str(PYTHON_BIN), "-m", "pip", "install", "-r", "requirements.txt"])
    logger.info("Dépendances installées.")


@task(name="format-code")
def format_code() -> None:
    """Formate les fichiers du pipeline avec black."""
    logger = get_run_logger()
    files = [str(path) for path in PROJECT_FILES]
    _run([str(PYTHON_BIN), "-m", "black", *files])
    logger.info("Formatage terminé.")


@task(name="code-quality")
def code_quality() -> None:
    """Vérifie le style et les erreurs avec flake8."""
    logger = get_run_logger()
    files = [str(path) for path in PROJECT_FILES]
    _run([str(PYTHON_BIN), "-m", "flake8", *files])
    logger.info("Qualité du code validée.")


@task(name="code-security")
def code_security() -> None:
    """Cherche les vulnérabilités de niveau moyen ou élevé avec bandit."""
    logger = get_run_logger()
    files = [str(path) for path in PROJECT_FILES]
    _run([str(PYTHON_BIN), "-m", "bandit", "-ll", "-q", *files])
    logger.info("Sécurité du code validée.")


@task(name="unit-tests")
def unit_tests() -> None:
    """Exécute la suite de tests unitaires."""
    logger = get_run_logger()
    _run([str(PYTHON_BIN), "-m", "pytest", "tests", "-q"])
    logger.info("Tests unitaires terminés.")


@task(name="prepare-data")
def prepare_data() -> None:
    """Lance la préparation des données."""
    logger = get_run_logger()
    _run([str(PYTHON_BIN), str(MAIN_PY), "--prepare"])
    logger.info("Préparation des données terminée.")


@task(name="train-model")
def train_model() -> None:
    """Lance l'entraînement du modèle."""
    logger = get_run_logger()
    _run([str(PYTHON_BIN), str(MAIN_PY), "--train"])
    logger.info("Entraînement terminé.")


@task(name="save-model")
def save_model() -> None:
    """Lance la sauvegarde du modèle entraîné."""
    logger = get_run_logger()
    _run([str(PYTHON_BIN), str(MAIN_PY), "--save"])
    logger.info("Modèle sauvegardé.")


@task(name="load-model")
def load_model() -> None:
    """Lance le chargement du modèle sauvegardé."""
    logger = get_run_logger()
    _run([str(PYTHON_BIN), str(MAIN_PY), "--load"])
    logger.info("Modèle chargé.")


@task(name="evaluate-model")
def evaluate_model() -> None:
    """Lance l'évaluation du modèle."""
    logger = get_run_logger()
    _run([str(PYTHON_BIN), str(MAIN_PY), "--evaluate"])
    logger.info("Évaluation terminée.")


@task(name="predict")
def predict() -> None:
    """Lance une prédiction de démonstration."""
    logger = get_run_logger()
    _run([str(PYTHON_BIN), str(MAIN_PY), "--predict"])
    logger.info("Prédiction terminée.")


@flow(name="code", log_prints=True)
def flow_code() -> None:
    """Vérifie le format, la qualité, la sécurité et les tests."""
    format_code()
    code_quality()
    code_security()
    unit_tests()


@flow(name="train", log_prints=True)
def flow_train() -> None:
    """Prépare les données puis entraîne le modèle."""
    prepare_data()
    train_model()


@flow(name="evaluate", log_prints=True)
def flow_evaluate() -> None:
    """Charge le modèle sauvegardé et l'évalue."""
    load_model()
    evaluate_model()


@flow(name="all", log_prints=True)
def flow_all() -> None:
    """Exécute les contrôles de code, puis le pipeline modèle."""
    fetch_project()
    install_dependencies()
    flow_code()
    prepare_data()
    train_model()
    save_model()
    evaluate_model()
    predict()


FLOWS = {
    "all": flow_all,
    "entrainement": flow_train,
    "train": flow_train,
    "evaluate": flow_evaluate,
    "code": flow_code,
}


def main() -> None:
    """Lance le flow demandé par --flow."""
    parser = argparse.ArgumentParser(description="Lance un flow Prefect.")
    parser.add_argument(
        "--flow",
        required=True,
        choices=sorted(FLOWS),
        help="Flow à exécuter.",
    )
    args = parser.parse_args()
    FLOWS[args.flow]()


if __name__ == "__main__":
    main()
