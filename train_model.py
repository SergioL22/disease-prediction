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
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("heart.csv")
ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "best_model.joblib"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"
BACKGROUND_PATH = ARTIFACTS_DIR / "X_background.csv"
TARGET_COL = "target"
RANDOM_STATE = 42
CV_FOLDS = 5
N_ITER = 20
BACKGROUND_SIZE = 100


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_COL}' in dataset")

    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    x = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return x, y


def build_search_spaces() -> dict[str, tuple[Pipeline, dict]]:
    return {
        "logistic_regression": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
            ]),
            {
                "model__C": [0.001, 0.01, 0.1, 1, 10, 100],
                "model__solver": ["lbfgs", "liblinear"],
            },
        ),
        "random_forest": (
            Pipeline([
                ("model", RandomForestClassifier(random_state=RANDOM_STATE)),
            ]),
            {
                "model__n_estimators": [100, 200, 300, 500],
                "model__max_depth": [3, 5, 10, 15, 20],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", "log2"],
            },
        ),
    }


def tune_and_evaluate(
    name: str,
    pipeline: Pipeline,
    param_dist: dict,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[Pipeline, dict]:
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=N_ITER,
        scoring="roc_auc",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
    )
    search.fit(x_train, y_train)

    best_pipeline = search.best_estimator_
    preds = best_pipeline.predict(x_test)
    probs = best_pipeline.predict_proba(x_test)[:, 1]

    metrics = {
        "model": name,
        "best_params": search.best_params_,
        "cv_roc_auc_mean": float(search.best_score_),
        "cv_roc_auc_std": float(search.cv_results_["std_test_score"][search.best_index_]),
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds)),
        "roc_auc": float(roc_auc_score(y_test, probs)),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        "classification_report": classification_report(y_test, preds, output_dict=True),
    }
    return best_pipeline, metrics


def print_model_metrics(metrics: dict[str, Any]) -> None:
    print(f"\nModel: {metrics['model']}")
    print(f"  CV ROC-AUC   : {metrics['cv_roc_auc_mean']:.4f} ± {metrics['cv_roc_auc_std']:.4f}")
    print(f"  Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Test F1      : {metrics['f1']:.4f}")
    print(f"  Test ROC-AUC : {metrics['roc_auc']:.4f}")
    print(f"  Best params  : {metrics['best_params']}")
    print(f"  Confusion    : {metrics['confusion_matrix']}")


def save_artifacts(
    best_model: Pipeline,
    feature_columns: list[str],
    results: list[dict],
    x_train: pd.DataFrame,
) -> None:
    best_result = max(results, key=lambda item: item["roc_auc"])

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)

    background = x_train.sample(min(BACKGROUND_SIZE, len(x_train)), random_state=RANDOM_STATE)
    background.to_csv(BACKGROUND_PATH, index=False)

    metadata = {
        "target_column": TARGET_COL,
        "feature_columns": feature_columns,
        "best_model": best_result["model"],
        "best_model_roc_auc": best_result["roc_auc"],
        "best_model_cv_roc_auc": best_result["cv_roc_auc_mean"],
        "best_model_cv_roc_auc_std": best_result["cv_roc_auc_std"],
        "all_results": results,
    }

    with METADATA_PATH.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nSaved artifacts:")
    print(f"- Model     : {MODEL_PATH}")
    print(f"- Metadata  : {METADATA_PATH}")
    print(f"- Background: {BACKGROUND_PATH}")


def main() -> None:
    df = load_data(DATA_PATH)
    x, y = split_features_target(df)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y,
    )

    search_spaces = build_search_spaces()
    results: list[dict] = []
    best_model: Pipeline | None = None
    best_roc_auc = -1.0

    for model_name, (pipeline, param_dist) in search_spaces.items():
        print(f"\nTuning {model_name}...")
        tuned_pipeline, model_metrics = tune_and_evaluate(
            model_name, pipeline, param_dist, x_train, y_train, x_test, y_test,
        )
        results.append(model_metrics)
        print_model_metrics(model_metrics)

        if model_metrics["roc_auc"] > best_roc_auc:
            best_roc_auc = model_metrics["roc_auc"]
            best_model = tuned_pipeline

    if best_model is None:
        raise RuntimeError("No model was trained.")

    save_artifacts(best_model, x.columns.tolist(), results, x_train)


if __name__ == "__main__":
    main()
