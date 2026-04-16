from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path("artifacts/best_model.joblib")
METADATA_PATH = Path("artifacts/metadata.json")

DEFAULT_INPUT = {
    "age": 57,
    "sex": 1,
    "cp": 2,
    "trestbps": 130,
    "chol": 236,
    "fbs": 0,
    "restecg": 1,
    "thalach": 174,
    "exang": 0,
    "oldpeak": 0.0,
    "slope": 2,
    "ca": 0,
    "thal": 2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Predict heart disease from patient features. "
            "Run without arguments to use a built-in example patient."
        )
    )

    parser.add_argument("--age", type=float, default=DEFAULT_INPUT["age"], help="Age in years")
    parser.add_argument("--sex", type=float, default=DEFAULT_INPUT["sex"], help="Sex encoded as in dataset")
    parser.add_argument("--cp", type=float, default=DEFAULT_INPUT["cp"], help="Chest pain type")
    parser.add_argument("--trestbps", type=float, default=DEFAULT_INPUT["trestbps"], help="Resting blood pressure")
    parser.add_argument("--chol", type=float, default=DEFAULT_INPUT["chol"], help="Serum cholesterol")
    parser.add_argument("--fbs", type=float, default=DEFAULT_INPUT["fbs"], help="Fasting blood sugar flag")
    parser.add_argument("--restecg", type=float, default=DEFAULT_INPUT["restecg"], help="Resting ECG result")
    parser.add_argument("--thalach", type=float, default=DEFAULT_INPUT["thalach"], help="Maximum heart rate achieved")
    parser.add_argument("--exang", type=float, default=DEFAULT_INPUT["exang"], help="Exercise induced angina flag")
    parser.add_argument("--oldpeak", type=float, default=DEFAULT_INPUT["oldpeak"], help="ST depression induced by exercise")
    parser.add_argument("--slope", type=float, default=DEFAULT_INPUT["slope"], help="Slope of peak exercise ST segment")
    parser.add_argument("--ca", type=float, default=DEFAULT_INPUT["ca"], help="Number of major vessels")
    parser.add_argument("--thal", type=float, default=DEFAULT_INPUT["thal"], help="Thalassemia encoding")

    return parser.parse_args()


def build_input_row(args: argparse.Namespace) -> dict[str, float]:
    return {
        "age": args.age,
        "sex": args.sex,
        "cp": args.cp,
        "trestbps": args.trestbps,
        "chol": args.chol,
        "fbs": args.fbs,
        "restecg": args.restecg,
        "thalach": args.thalach,
        "exang": args.exang,
        "oldpeak": args.oldpeak,
        "slope": args.slope,
        "ca": args.ca,
        "thal": args.thal,
    }


def main() -> None:
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError(
            "Model artifacts not found. Run 'python train_model.py' first."
        )

    model = joblib.load(MODEL_PATH)

    with METADATA_PATH.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    feature_columns = metadata["feature_columns"]
    args = parse_args()
    row = build_input_row(args)

    missing = [col for col in feature_columns if col not in row]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    x_input = pd.DataFrame([[row[col] for col in feature_columns]], columns=feature_columns)

    pred = int(model.predict(x_input)[0])
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(x_input)[0][1])
    else:
        probability = float(pred)

    print(f"Predicted class: {pred}")
    print(f"Predicted disease probability: {probability:.4f}")


if __name__ == "__main__":
    main()
