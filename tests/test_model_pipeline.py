"""Tests unitaires des fonctions du pipeline ML."""

import pandas as pd
import pytest
from sklearn.datasets import load_iris

from model_pipeline import (
    evaluate_model,
    load_model,
    prepare_data,
    save_model,
    train_model,
    validate_data,
)


def _iris_frame() -> pd.DataFrame:
    iris = load_iris(as_frame=True)
    frame = iris.frame.copy()
    frame.columns = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "species",
    ]
    frame["species"] = frame["species"].map(dict(enumerate(iris.target_names)))
    return frame


def test_validate_data_rejects_missing_target() -> None:
    frame = _iris_frame().drop(columns=["species"])
    with pytest.raises(ValueError, match="Colonnes manquantes"):
        validate_data(frame)


def test_validate_data_rejects_missing_values() -> None:
    frame = _iris_frame()
    frame.loc[0, "sepal_length"] = None
    with pytest.raises(ValueError, match="valeurs manquantes"):
        validate_data(frame)


def test_load_model_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "absent.joblib")


def test_prepare_train_save_load_evaluate(tmp_path) -> None:
    csv_path = tmp_path / "iris.csv"
    _iris_frame().to_csv(csv_path, index=False)
    prepared = prepare_data(
        csv_path=csv_path,
        output_dir=tmp_path / "processed",
        test_size=0.2,
        random_state=42,
    )

    assert prepared["n_train"] == 120
    assert prepared["n_test"] == 30

    model = train_model(prepared["X_train"], prepared["y_train"])
    model_path = save_model(model, tmp_path / "model.joblib")
    loaded = load_model(model_path)
    metrics = evaluate_model(loaded, prepared["X_test"], prepared["y_test"])

    assert metrics["accuracy"] >= 0.8
    assert metrics["f1_macro"] >= 0.8
    assert list(loaded.named_steps) == ["scaler", "classifier"]
