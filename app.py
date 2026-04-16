from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path("artifacts/best_model.joblib")
METADATA_PATH = Path("artifacts/metadata.json")
TARGET_LABELS = {0: "Lower disease likelihood", 1: "Higher disease likelihood"}

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

FEATURE_INFO: dict[str, dict[str, Any]] = {
    "age": {"label": "Age", "min": 1, "max": 120, "step": 1},
    "sex": {"label": "Sex", "options": {0: "Female", 1: "Male"}},
    "cp": {"label": "Chest pain type", "options": {0: "Typical angina", 1: "Atypical angina", 2: "Non-anginal pain", 3: "Asymptomatic"}},
    "trestbps": {"label": "Resting blood pressure (mm Hg)", "min": 80, "max": 220, "step": 1},
    "chol": {"label": "Serum cholesterol (mg/dL)", "min": 100, "max": 600, "step": 1},
    "fbs": {"label": "Fasting blood sugar > 120 mg/dL", "options": {0: "No", 1: "Yes"}},
    "restecg": {"label": "Resting ECG result", "options": {0: "Normal", 1: "ST-T wave abnormality", 2: "Left ventricular hypertrophy"}},
    "thalach": {"label": "Max heart rate achieved", "min": 60, "max": 220, "step": 1},
    "exang": {"label": "Exercise induced angina", "options": {0: "No", 1: "Yes"}},
    "oldpeak": {"label": "ST depression induced by exercise", "min": 0.0, "max": 10.0, "step": 0.1},
    "slope": {"label": "Slope of peak exercise ST segment", "options": {0: "Upsloping", 1: "Flat", 2: "Downsloping"}},
    "ca": {"label": "Number of major vessels (0-3)", "min": 0, "max": 3, "step": 1},
    "thal": {"label": "Thalassemia", "options": {1: "Normal", 2: "Fixed defect", 3: "Reversible defect"}},
}


def load_artifacts() -> tuple[Any, dict[str, Any]]:
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError(
            "Model artifacts not found. Run 'python train_model.py' first, then refresh this app."
        )

    model = joblib.load(MODEL_PATH)
    metadata_text = METADATA_PATH.read_text(encoding="utf-8")
    metadata = json.loads(metadata_text)
    return model, metadata


@st.cache_data
def load_model() -> tuple[Any, dict[str, Any]]:
    return load_artifacts()


def build_input_dataframe(values: dict[str, float], feature_columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame([[values[column] for column in feature_columns]], columns=feature_columns)


def predict(model: Any, x_input: pd.DataFrame) -> tuple[int, float]:
    predicted_class = int(model.predict(x_input)[0])
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(x_input)[0][1])
    else:
        probability = float(predicted_class)
    return predicted_class, probability


def render_sidebar_inputs() -> dict[str, float]:
    values: dict[str, float] = {}

    st.sidebar.header("Patient features")
    st.sidebar.write("Adjust the values and click Predict.")

    for feature, info in FEATURE_INFO.items():
        if "options" in info:
            options = list(info["options"].items())
            labels = [label for _, label in options]
            values_list = [value for value, _ in options]
            selected_index = values_list.index(DEFAULT_INPUT[feature]) if DEFAULT_INPUT[feature] in values_list else 0
            selection = st.sidebar.selectbox(info["label"], labels, index=selected_index)
            values[feature] = values_list[labels.index(selection)]
        elif isinstance(info.get("step"), float):
            values[feature] = st.sidebar.number_input(
                info["label"],
                min_value=info["min"],
                max_value=info["max"],
                value=float(DEFAULT_INPUT[feature]),
                step=info["step"],
                format="%.1f",
            )
        else:
            values[feature] = st.sidebar.number_input(
                info["label"],
                min_value=info["min"],
                max_value=info["max"],
                value=int(DEFAULT_INPUT[feature]),
                step=int(info["step"]),
            )

    return values


def main() -> None:
    st.set_page_config(page_title="Heart Disease Predictor", layout="wide")
    st.title("Heart Disease Prediction")
    st.write(
        "Use this interface to explore how patient features affect the predicted heart disease likelihood. "
        "The model was trained on the `heart.csv` dataset and uses the best saved model from `artifacts/`."
    )

    try:
        model, metadata = load_model()
    except FileNotFoundError as error:
        st.error(str(error))
        return

    sidebar_values = render_sidebar_inputs()

    if st.sidebar.button("Predict"):
        x_input = build_input_dataframe(sidebar_values, metadata["feature_columns"])
        predicted_class, probability = predict(model, x_input)

        st.subheader("Prediction result")
        st.metric("Predicted class", f"{predicted_class} ({TARGET_LABELS[predicted_class]})")
        st.metric("Probability of disease", f"{probability:.4f}")

        with st.expander("Input feature values"):
            st.write(x_input.T.rename(columns={0: "value"}))

        st.subheader("Model information")
        st.write(f"**Best model:** {metadata.get('best_model', 'unknown')}  ")
        st.write(f"**Best ROC-AUC:** {metadata.get('best_model_roc_auc', 'unknown'):.4f}")

        if metadata.get("all_results"):
            st.subheader("Training evaluation results")
            results_df = pd.DataFrame(metadata["all_results"])
            display_columns = ["model", "accuracy", "f1", "roc_auc"]
            st.dataframe(results_df[display_columns])
    else:
        st.info("Click Predict in the sidebar to compute a prediction.")

    with st.expander("Training metadata and feature columns"):
        st.write("Metadata loaded from `artifacts/metadata.json`.")
        st.json(metadata)


if __name__ == "__main__":
    main()
