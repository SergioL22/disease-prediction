from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from train_model import (
    build_search_spaces,
    load_data,
    save_artifacts,
    split_features_target,
    tune_and_evaluate,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Minimal heart-disease-shaped DataFrame for fast tests."""
    return pd.DataFrame(
        {
            "age": [50, 60, 45, 55, 40, 65, 52, 48, 58, 43,
                    50, 60, 45, 55, 40, 65, 52, 48, 58, 43],
            "sex": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
                    0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "chol": [200, 220, 180, 240, 190, 260, 210, 195, 230, 175,
                     200, 220, 180, 240, 190, 260, 210, 195, 230, 175],
            "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
                       1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        }
    )


# --- load_data ---

def test_load_data_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_data(tmp_path / "nonexistent.csv")


def test_load_data_missing_target_column(tmp_path: Path) -> None:
    csv = tmp_path / "data.csv"
    csv.write_text("age,sex\n50,1\n60,0\n")
    with pytest.raises(ValueError, match="target"):
        load_data(csv)


def test_load_data_returns_dataframe(tmp_path: Path, sample_df: pd.DataFrame) -> None:
    csv = tmp_path / "data.csv"
    sample_df.to_csv(csv, index=False)
    df = load_data(csv)
    assert isinstance(df, pd.DataFrame)
    assert "target" in df.columns
    assert len(df) == len(sample_df)


# --- split_features_target ---

def test_split_features_target_shapes(sample_df: pd.DataFrame) -> None:
    x, y = split_features_target(sample_df)
    assert x.shape == (len(sample_df), sample_df.shape[1] - 1)
    assert len(y) == len(sample_df)


def test_split_features_target_no_target_in_x(sample_df: pd.DataFrame) -> None:
    x, _ = split_features_target(sample_df)
    assert "target" not in x.columns


def test_split_features_target_y_is_target(sample_df: pd.DataFrame) -> None:
    _, y = split_features_target(sample_df)
    assert y.name == "target"
    assert list(y) == list(sample_df["target"])


# --- build_search_spaces ---

def test_build_search_spaces_returns_expected_keys() -> None:
    spaces = build_search_spaces()
    assert set(spaces.keys()) == {"logistic_regression", "random_forest"}


def test_build_search_spaces_pipelines_are_valid() -> None:
    spaces = build_search_spaces()
    for pipeline, param_dist in spaces.values():
        assert isinstance(pipeline, Pipeline)
        assert isinstance(param_dist, dict)
        assert all(k.startswith("model__") for k in param_dist)


# --- save_artifacts ---

def test_save_artifacts_creates_files(tmp_path: Path, sample_df: pd.DataFrame, monkeypatch) -> None:
    import train_model
    monkeypatch.setattr(train_model, "ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr(train_model, "MODEL_PATH", tmp_path / "best_model.joblib")
    monkeypatch.setattr(train_model, "METADATA_PATH", tmp_path / "metadata.json")
    monkeypatch.setattr(train_model, "BACKGROUND_PATH", tmp_path / "X_background.csv")

    spaces = build_search_spaces()
    lr_pipeline, _ = spaces["logistic_regression"]
    x, y = split_features_target(sample_df)
    lr_pipeline.fit(x, y)

    results = [
        {
            "model": "logistic_regression",
            "best_params": {"model__C": 1},
            "cv_roc_auc_mean": 0.85,
            "cv_roc_auc_std": 0.03,
            "accuracy": 0.80,
            "f1": 0.80,
            "roc_auc": 0.85,
            "confusion_matrix": [[4, 1], [1, 4]],
            "classification_report": {},
        }
    ]
    save_artifacts(lr_pipeline, x.columns.tolist(), results, x)

    assert (tmp_path / "best_model.joblib").exists()
    assert (tmp_path / "X_background.csv").exists()
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["best_model"] == "logistic_regression"
    assert metadata["feature_columns"] == x.columns.tolist()
    assert "best_model_roc_auc" in metadata
    assert "best_model_cv_roc_auc" in metadata


# --- tune_and_evaluate (lightweight integration) ---

def test_tune_and_evaluate_returns_expected_metric_keys(sample_df: pd.DataFrame) -> None:
    spaces = build_search_spaces()
    pipeline, param_dist = spaces["logistic_regression"]
    # Narrow the search to keep the test fast
    param_dist = {"model__C": [1.0], "model__solver": ["lbfgs"]}

    x, y = split_features_target(sample_df)
    split = len(x) // 5
    x_train, x_test = x.iloc[split:], x.iloc[:split]
    y_train, y_test = y.iloc[split:], y.iloc[:split]

    _, metrics = tune_and_evaluate(
        "logistic_regression", pipeline, param_dist, x_train, y_train, x_test, y_test
    )
    expected_keys = {"model", "best_params", "cv_roc_auc_mean", "cv_roc_auc_std",
                     "accuracy", "f1", "roc_auc", "confusion_matrix", "classification_report"}
    assert expected_keys.issubset(metrics.keys())
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["accuracy"] <= 1.0
