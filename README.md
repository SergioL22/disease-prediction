# Heart Disease Prediction

A Python project for training and deploying a heart disease prediction model from patient data.
It includes both a command-line script and a Streamlit web app for interactive prediction.

The model is trained on the [UCI Heart Disease dataset (Cleveland)](https://archive.ics.uci.edu/dataset/45/heart+disease)
and uses 13 clinical features to predict the likelihood of heart disease in a patient.

---

## Project structure

```
.
├── train_model.py        # Train, tune, and save the best model
├── app.py                # Streamlit web app for interactive prediction
├── predict.py            # Command-line inference script
├── heart.csv             # Source dataset (1,025 patients, 13 features + target)
├── artifacts/            # Saved model, metadata, and SHAP background (created by train_model.py)
├── tests/                # Unit tests (pytest)
├── requirements.txt      # Pinned runtime dependencies
└── requirements-dev.txt  # Adds pytest for running tests
```

---

## Setup

**1. Create and activate a virtual environment**

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

**2. Install dependencies**

```bash
# Runtime only
pip install -r requirements.txt

# Runtime + testing tools
pip install -r requirements-dev.txt
```

---

## How to use

### Train the model

```bash
python train_model.py
```

Trains Logistic Regression and Random Forest models, each tuned with
`RandomizedSearchCV` (5-fold stratified CV, 20 iterations).
Saves the best model to `artifacts/best_model.joblib`, evaluation metrics to
`artifacts/metadata.json`, and a background sample for SHAP explanations to
`artifacts/X_background.csv`.

### Interactive web app

```bash
streamlit run app.py
```

Opens a browser interface where you can adjust patient features in the sidebar and get:

- A **colour-coded risk level** (Low / Moderate / High)
- A **plain-language summary** explaining the top reasons in everyday terms
- A **per-patient SHAP chart** showing exactly which features pushed the score up or down
- A **global feature importance chart** showing what the model generally relies on

### Command-line inference

```bash
# Run with the built-in example patient
python predict.py

# Supply your own values
python predict.py --age 63 --sex 1 --cp 3 --trestbps 145 --chol 233 \
  --fbs 1 --restecg 0 --thalach 150 --exang 0 --oldpeak 2.3 \
  --slope 0 --ca 0 --thal 1
```

---

## Running tests

```bash
python -m pytest tests/ -v
```

---

## Dataset features

The dataset contains 1,025 patient records drawn from four clinical centers
(Cleveland, Hungary, Switzerland, VA Long Beach). Each row has 13 input features and one target label.

| Feature    | Description                                  | Values / Range                                                                 |
|------------|----------------------------------------------|--------------------------------------------------------------------------------|
| `age`      | Age in years                                 | Integer (29 – 77)                                                              |
| `sex`      | Biological sex                               | 0 = Female, 1 = Male                                                           |
| `cp`       | Chest pain type                              | 0 = Typical angina, 1 = Atypical angina, 2 = Non-anginal pain, 3 = Asymptomatic |
| `trestbps` | Resting blood pressure (mm Hg)               | Integer (94 – 200)                                                             |
| `chol`     | Serum cholesterol (mg/dL)                    | Integer (126 – 564)                                                            |
| `fbs`      | Fasting blood sugar > 120 mg/dL              | 0 = No, 1 = Yes                                                                |
| `restecg`  | Resting ECG results                          | 0 = Normal, 1 = ST-T wave abnormality, 2 = Left ventricular hypertrophy       |
| `thalach`  | Maximum heart rate achieved                  | Integer (71 – 202)                                                             |
| `exang`    | Exercise-induced angina                      | 0 = No, 1 = Yes                                                                |
| `oldpeak`  | ST depression induced by exercise vs. rest   | Float (0.0 – 6.2)                                                              |
| `slope`    | Slope of peak exercise ST segment            | 0 = Upsloping, 1 = Flat, 2 = Downsloping                                      |
| `ca`       | Number of major vessels colored by fluoroscopy | 0 – 3                                                                        |
| `thal`     | Thalassemia type                             | 1 = Normal, 2 = Fixed defect, 3 = Reversible defect                           |

**Target:** `0` = lower disease likelihood, `1` = higher disease likelihood.

---

## Model approach

Two scikit-learn pipelines are trained and compared:

| Model               | Preprocessing   | Tuned hyperparameters                                                          |
|---------------------|-----------------|--------------------------------------------------------------------------------|
| Logistic Regression | StandardScaler  | `C`, `solver`                                                                  |
| Random Forest       | None            | `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features` |

Both are tuned with `RandomizedSearchCV` (20 iterations, 5-fold stratified CV, scoring = ROC-AUC).
The model with the highest held-out ROC-AUC is saved as the production artifact.

SHAP explanations use `TreeExplainer` for Random Forest and `LinearExplainer` for Logistic Regression,
giving exact per-prediction feature contributions rather than approximations.

---

## Dependencies

| Package       | Version |
|---------------|---------|
| pandas        | 3.0.1   |
| scikit-learn  | 1.8.0   |
| joblib        | 1.5.3   |
| numpy         | 2.4.3   |
| shap          | 0.52.0  |
| streamlit     | 1.58.0  |
