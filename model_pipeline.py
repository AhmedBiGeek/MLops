"""Fonctions réutilisables du pipeline de classification Iris."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "iris.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
STAGING_MODEL_PATH = ROOT / "models" / "model_trained.joblib"
MODEL_PATH = ROOT / "models" / "model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"

FEATURE_COLUMNS = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]
TARGET_COLUMN = "species"
REQUIRED_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]


def validate_data(df: pd.DataFrame) -> None:
    """Vérifie que le jeu de données est complet et exploitable.

    Args:
        df: Tableau chargé depuis le CSV source.

    Raises:
        ValueError: Si une règle de qualité des données n'est pas respectée.
    """
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes: {', '.join(missing)}")
    if df.empty:
        raise ValueError("Le jeu de données est vide.")
    if df[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError("Des valeurs manquantes sont présentes.")
    features = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if features.isna().any().any():
        raise ValueError("Les variables explicatives doivent être numériques.")
    if df[TARGET_COLUMN].nunique() < 2:
        raise ValueError("La cible doit contenir au moins deux classes.")


def prepare_data(
    csv_path: Path = DATA_PATH,
    output_dir: Path = PROCESSED_DIR,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """Charge, valide et découpe les données en ensembles train et test.

    Args:
        csv_path: Chemin du CSV source.
        output_dir: Dossier où écrire les fichiers découpés.
        test_size: Proportion réservée au test.
        random_state: Graine du découpage.

    Returns:
        Dictionnaire avec les tailles et les quatre tableaux.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Fichier de données introuvable: {csv_path}")

    df = pd.read_csv(csv_path)
    validate_data(df)
    features = df[FEATURE_COLUMNS].apply(pd.to_numeric)
    target = df[TARGET_COLUMN]
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    x_train.to_csv(output_dir / "X_train.csv", index=False)
    x_test.to_csv(output_dir / "X_test.csv", index=False)
    y_train.to_csv(output_dir / "y_train.csv", index=False)
    y_test.to_csv(output_dir / "y_test.csv", index=False)
    return {
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "X_train": x_train,
        "X_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
    }


def load_splits(
    processed_dir: Path = PROCESSED_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Recharge les ensembles train et test écrits par prepare_data.

    Args:
        processed_dir: Dossier produit par prepare_data.

    Returns:
        X_train, X_test, y_train, y_test.
    """
    processed_dir = Path(processed_dir)
    x_train = pd.read_csv(processed_dir / "X_train.csv")
    x_test = pd.read_csv(processed_dir / "X_test.csv")
    y_train = pd.read_csv(processed_dir / "y_train.csv")[TARGET_COLUMN]
    y_test = pd.read_csv(processed_dir / "y_test.csv")[TARGET_COLUMN]
    return x_train, x_test, y_train, y_test


def train_model(x_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Entraîne une régression logistique standardisée.

    Args:
        x_train: Variables explicatives d'entraînement.
        y_train: Cible d'entraînement.

    Returns:
        Pipeline sklearn ajusté.
    """
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(max_iter=200),
            ),
        ]
    )
    model.fit(x_train, y_train)
    return model


def evaluate_model(
    model: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Mesure l'accuracy et le F1 macro sur le jeu de test.

    Args:
        model: Modèle déjà entraîné.
        x_test: Variables explicatives de test.
        y_test: Cible de test.

    Returns:
        Dictionnaire des métriques.
    """
    predictions = model.predict(x_test)
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1_macro": float(f1_score(y_test, predictions, average="macro")),
    }


def save_model(model: Pipeline, path: Path = MODEL_PATH) -> Path:
    """Sauvegarde le modèle avec joblib.

    Args:
        model: Modèle à sérialiser.
        path: Destination du fichier joblib.

    Returns:
        Chemin du fichier écrit.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path: Path = MODEL_PATH) -> Pipeline:
    """Charge un modèle joblib.

    Args:
        path: Fichier joblib à lire.

    Returns:
        Modèle chargé.

    Raises:
        FileNotFoundError: Si le fichier n'existe pas.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Modèle introuvable: {path}")
    return joblib.load(path)


def predict(model: Pipeline, features: pd.DataFrame) -> pd.DataFrame:
    """Prédit la classe et la confiance pour chaque ligne.

    Args:
        model: Modèle chargé.
        features: Tableau avec les colonnes explicatives.

    Returns:
        Tableau avec les colonnes species et confidence.
    """
    labels = model.predict(features)
    confidence = model.predict_proba(features).max(axis=1)
    return pd.DataFrame({"species": labels, "confidence": confidence})
