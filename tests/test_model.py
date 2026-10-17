from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from starter.ml.data import process_data
from starter.ml.model import (
    compute_model_metrics, inference, load, save, train_model,
)

CAT_FEATURES = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country",
]
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "census.csv"


@pytest.fixture(scope="module")
def processed():
    # A small sample keeps the tests fast.
    df = pd.read_csv(DATA_PATH, skipinitialspace=True).sample(
        500, random_state=0
    )
    X, y, encoder, lb = process_data(
        df, categorical_features=CAT_FEATURES, label="salary", training=True
    )
    return df, X, y, encoder, lb


def test_train_model_returns_fitted_forest(processed):
    _, X, y, _, _ = processed
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)
    assert hasattr(model, "classes_")


def test_inference_returns_binary_prediction_per_row(processed):
    _, X, y, _, _ = processed
    preds = inference(train_model(X, y), X)
    assert isinstance(preds, np.ndarray)
    assert len(preds) == len(X)
    assert set(preds) <= {0, 1}


def test_compute_model_metrics_known_values():
    y = np.array([1, 1, 0, 0])
    preds = np.array([1, 0, 1, 0])
    assert compute_model_metrics(y, preds) == (0.5, 0.5, 0.5)


def test_process_data_inference_matches_training_columns(processed):
    df, X, _, encoder, lb = processed
    X_inf, _, _, _ = process_data(
        df.drop(columns="salary"), categorical_features=CAT_FEATURES,
        training=False, encoder=encoder, lb=lb,
    )
    assert X_inf.shape == X.shape


def test_save_load_roundtrip(processed, tmp_path):
    _, X, y, _, _ = processed
    model = train_model(X, y)
    save(model, tmp_path / "model.joblib")
    loaded = load(tmp_path / "model.joblib")
    assert np.array_equal(inference(loaded, X), inference(model, X))
