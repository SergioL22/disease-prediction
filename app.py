from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st

MODEL_PATH = Path("artifacts/best_model.joblib")
METADATA_PATH = Path("artifacts/metadata.json")
BACKGROUND_PATH = Path("artifacts/X_background.csv")
TARGET_LABELS = {0: "Lower disease likelihood", 1: "Higher disease likelihood"}

RISK_LEVELS = [
    (0.40, "Low", "success", "The model predicts a lower likelihood of heart disease."),
    (0.65, "Moderate", "warning", "The model predicts a moderate likelihood of heart disease."),
    (1.01, "High", "error", "The model predicts a higher likelihood of heart disease."),
]

FEATURE_PLAIN_LABEL: dict[str, str] = {
    "age":      "your age",
    "sex":      "your sex",
    "cp":       "your chest pain type",
    "trestbps": "your resting blood pressure",
    "chol":     "your cholesterol level",
    "fbs":      "your fasting blood sugar",
    "restecg":  "your resting heart rhythm (ECG)",
    "thalach":  "your maximum heart rate",
    "exang":    "chest pain during exercise",
    "oldpeak":  "how your heart copes under exercise stress",
    "slope":    "the pattern of your heart signal during exercise",
    "ca":       "the number of blocked heart vessels",
    "thal":     "your thalassemia status (a blood condition)",
}

FEATURE_PLAIN_HIGH: dict[str, str] = {
    "age":      "older age is an established risk factor for heart disease",
    "sex":      "men tend to develop heart disease more often in this dataset",
    "cp":       "this chest pain pattern is associated with higher heart disease risk",
    "trestbps": "higher resting blood pressure puts extra strain on the heart",
    "chol":     "elevated cholesterol can build up in artery walls and restrict blood flow",
    "fbs":      "high blood sugar (a sign of diabetes) raises the risk of heart problems",
    "restecg":  "abnormal heart activity at rest may indicate an underlying issue",
    "thalach":  "a lower peak heart rate can suggest reduced heart capacity",
    "exang":    "chest pain during physical activity is a warning sign of restricted blood flow",
    "oldpeak":  "a large dip in the heart signal under stress suggests the heart is struggling",
    "slope":    "a flat or downward-sloping heart signal during exercise can indicate poor blood supply",
    "ca":       "more blocked vessels means less blood can reach the heart muscle",
    "thal":     "this blood condition type is linked to a higher rate of heart complications",
}

FEATURE_PLAIN_LOW: dict[str, str] = {
    "age":      "younger age is a protective factor against heart disease",
    "sex":      "women tend to have lower heart disease rates in this dataset",
    "cp":       "this chest pain pattern is less commonly linked to heart disease",
    "trestbps": "normal resting blood pressure suggests the heart is not under excess strain",
    "chol":     "your cholesterol level is less likely to be causing arterial blockage",
    "fbs":      "normal blood sugar reduces diabetes-related heart risk",
    "restecg":  "a normal resting heart rhythm is a reassuring sign",
    "thalach":  "a high peak heart rate indicates good heart reserve and capacity",
    "exang":    "no chest pain during exercise suggests blood is flowing well",
    "oldpeak":  "little change in the heart signal under stress suggests the heart copes well",
    "slope":    "an upward-sloping heart signal during exercise is the healthiest pattern",
    "ca":       "no blocked vessels means blood can flow freely to the heart",
    "thal":     "a normal thalassemia result is associated with lower heart risk",
}

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
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return model, metadata


@st.cache_resource
def load_model() -> tuple[Any, dict[str, Any]]:
    return load_artifacts()


def build_input_dataframe(values: dict[str, float], feature_columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame([[values[col] for col in feature_columns]], columns=feature_columns)


def run_prediction(model: Any, x_input: pd.DataFrame) -> tuple[int, float]:
    predicted_class = int(model.predict(x_input)[0])
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(x_input)[0][1])
    else:
        probability = float(predicted_class)
    return predicted_class, probability


def get_feature_importances(model: Any, feature_columns: list[str]) -> pd.Series:
    """Extract normalised feature importances from a fitted sklearn Pipeline."""
    step = model.named_steps["model"]
    if hasattr(step, "feature_importances_"):
        scores = step.feature_importances_
    elif hasattr(step, "coef_"):
        scores = np.abs(step.coef_[0])
    else:
        return pd.Series(dtype=float)

    series = pd.Series(scores, index=feature_columns)
    total = series.sum()
    return (series / total).sort_values(ascending=True) if total > 0 else series.sort_values(ascending=True)


def render_risk_result(predicted_class: int, probability: float) -> None:
    for threshold, level, alert_type, message in RISK_LEVELS:
        if probability < threshold:
            getattr(st, alert_type)(f"**{level} risk** — {message}")
            break

    col1, col2 = st.columns(2)
    col1.metric("Predicted class", f"{predicted_class} — {TARGET_LABELS[predicted_class]}")
    col2.metric("Disease probability", f"{probability:.1%}")
    st.progress(probability, text=f"{probability:.1%} probability of heart disease")


def render_feature_importances(model: Any, feature_columns: list[str]) -> None:
    importances = get_feature_importances(model, feature_columns)
    if importances.empty:
        return

    st.subheader("Feature importances")
    st.caption(
        "How much each feature contributed to this model's decisions. "
        "For Random Forest this reflects impurity reduction; for Logistic Regression it reflects coefficient magnitude."
    )
    st.bar_chart(importances, horizontal=True)


@st.cache_resource
def load_shap_explainer(_model: Any, model_name: str) -> tuple[Any, pd.DataFrame]:
    """Create and cache a SHAP explainer. Underscore on _model skips Streamlit hashing."""
    background = pd.read_csv(BACKGROUND_PATH)
    step = _model.named_steps["model"]

    if model_name == "random_forest":
        explainer = shap.TreeExplainer(step)
    else:
        background_scaled = pd.DataFrame(
            _model[:-1].transform(background), columns=background.columns
        )
        explainer = shap.LinearExplainer(step, background_scaled)
        background = background_scaled

    return explainer, background


def compute_shap_series(model: Any, model_name: str, x_input: pd.DataFrame) -> pd.Series:
    explainer, _ = load_shap_explainer(model, model_name)

    if model_name == "random_forest":
        x_shap = x_input
    else:
        x_shap = pd.DataFrame(model[:-1].transform(x_input), columns=x_input.columns)

    explanation = explainer(x_shap)
    vals = explanation.values

    # TreeExplainer returns [n_samples, n_features, n_classes]; take class 1 (disease)
    # LinearExplainer returns [n_samples, n_features] for the positive class directly
    if vals.ndim == 3:
        vals = vals[0, :, 1]
    else:
        vals = vals[0]

    return pd.Series(vals, index=x_input.columns)


def render_plain_summary(shap_series: pd.Series) -> None:
    top3 = shap_series.reindex(shap_series.abs().sort_values(ascending=False).index).head(3)
    raising  = [(f, v) for f, v in top3.items() if v > 0]
    lowering = [(f, v) for f, v in top3.items() if v <= 0]

    sentences: list[str] = []

    if raising:
        factors = ", ".join(
            f"**{FEATURE_PLAIN_LABEL[f]}** ({FEATURE_PLAIN_HIGH[f]})"
            for f, _ in raising
        )
        sentences.append(f"The main concern{'s are' if len(raising) > 1 else ' is'} {factors}.")

    if lowering:
        factors = " and ".join(
            f"**{FEATURE_PLAIN_LABEL[f]}** ({FEATURE_PLAIN_LOW[f]})"
            for f, _ in lowering
        )
        sentences.append(f"Working in your favour: {factors}.")

    if sentences:
        st.markdown(" ".join(sentences))


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


def render_training_summary(metadata: dict[str, Any]) -> None:
    st.subheader("Model information")
    col1, col2, col3 = st.columns(3)
    col1.metric("Best model", metadata.get("best_model", "unknown"))
    col2.metric("Test ROC-AUC", f"{metadata.get('best_model_roc_auc', 0):.4f}")
    cv_mean = metadata.get("best_model_cv_roc_auc")
    cv_std = metadata.get("best_model_cv_roc_auc_std")
    if cv_mean is not None:
        label = f"{cv_mean:.4f} ± {cv_std:.4f}" if cv_std is not None else f"{cv_mean:.4f}"
        col3.metric("CV ROC-AUC (5-fold)", label)

    if metadata.get("all_results"):
        st.subheader("Training evaluation results")
        results_df = pd.DataFrame(metadata["all_results"])
        display_cols = [c for c in ["model", "cv_roc_auc_mean", "cv_roc_auc_std", "accuracy", "f1", "roc_auc"] if c in results_df.columns]
        st.dataframe(
            results_df[display_cols].rename(columns={
                "cv_roc_auc_mean": "CV ROC-AUC (mean)",
                "cv_roc_auc_std": "CV ROC-AUC (std)",
                "roc_auc": "Test ROC-AUC",
                "accuracy": "Test Accuracy",
                "f1": "Test F1",
            }),
            hide_index=True,
        )


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
        predicted_class, probability = run_prediction(model, x_input)

        st.subheader("Prediction result")
        render_risk_result(predicted_class, probability)

        if BACKGROUND_PATH.exists():
            shap_series = compute_shap_series(model, metadata["best_model"], x_input)
            render_plain_summary(shap_series)
            st.subheader("Why this prediction?")
            st.caption(
                "Each bar shows how much a feature pushed the predicted disease probability up (positive) "
                "or down (negative) for this specific patient."
            )
            sorted_series = shap_series.reindex(shap_series.abs().sort_values().index)
            st.bar_chart(sorted_series, horizontal=True)

        render_feature_importances(model, metadata["feature_columns"])

        with st.expander("Input feature values"):
            st.dataframe(x_input.T.rename(columns={0: "value"}))

        render_training_summary(metadata)

    else:
        st.info("Click Predict in the sidebar to compute a prediction.")
        render_training_summary(metadata)

    with st.expander("Raw training metadata"):
        st.json(metadata)


if __name__ == "__main__":
    main()
