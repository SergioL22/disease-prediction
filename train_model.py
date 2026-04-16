from __future__ import annotations

from pathlib import Path
import json
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("heart.csv")
ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "best_model.joblib"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"
TARGET_COL = "target"
RANDOM_STATE = 42


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_COL}' in dataset")

    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split the dataframe into model features (X) and target labels (y)."""
    x = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return x, y


def build_models() -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=None,
                        min_samples_split=2,
                        min_samples_leaf=1,
                        random_state=RANDOM_STATE,
                    ),
                )
            ]
        ),
    }


def evaluate_model(name: str, model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    preds = model.predict(x_test)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x_test)[:, 1]
    else:
        probs = preds

    metrics = {
        "model": name,
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds)),
        "roc_auc": float(roc_auc_score(y_test, probs)),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        "classification_report": classification_report(y_test, preds, output_dict=True),
    }
    return metrics


def print_model_metrics(metrics: dict[str, Any]) -> None:
    """Print model metrics in a compact, readable format."""
    print(f"\nModel: {metrics['model']}")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  F1 Score : {metrics['f1']:.4f}")
    print(f"  ROC-AUC  : {metrics['roc_auc']:.4f}")
    print(f"  Confusion: {metrics['confusion_matrix']}")


def save_artifacts(best_model: Pipeline, feature_columns: list[str], results: list[dict]) -> None:
    """Persist trained model and metadata used by prediction script."""
    best_result = max(results, key=lambda item: item["roc_auc"])

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)

    metadata = {
        "target_column": TARGET_COL,
        "feature_columns": feature_columns,
        "best_model": best_result["model"],
        "best_model_roc_auc": best_result["roc_auc"],
        "all_results": results,
    }

    with METADATA_PATH.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nSaved artifacts:")
    print(f"- Model: {MODEL_PATH}")
    print(f"- Metadata: {METADATA_PATH}")


def main() -> None:
    df = load_data(DATA_PATH)
    x, y = split_features_target(df)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    models = build_models()
    results: list[dict] = []

    best_model: Pipeline | None = None
    best_roc_auc = -1.0

    for model_name, pipeline in models.items():
        pipeline.fit(x_train, y_train)
        model_metrics = evaluate_model(model_name, pipeline, x_test, y_test)
        results.append(model_metrics)
        print_model_metrics(model_metrics)

        if model_metrics["roc_auc"] > best_roc_auc:
            best_roc_auc = model_metrics["roc_auc"]
            best_model = pipeline

    if best_model is None:
        raise RuntimeError("No model was trained.")

    save_artifacts(best_model, x.columns.tolist(), results)


if __name__ == "__main__":
    main()
