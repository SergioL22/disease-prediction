# Disease Prediction

This project is a work in progress. It is built to train and deploy a heart disease prediction model using the `heart.csv` dataset.

## Current contents

- `train_model.py` — trains candidate models, evaluates them, and saves the best model and metadata.
- `app.py` — Streamlit application for interactive prediction using saved model artifacts.
- `predict.py` — command-line inference script for single-patient predictions.
- `artifacts/` — saved model and metadata used by predictions.

## Status

The project is functional for training and inference, but still needs improvements for production readiness and better user experience.

## Planned improvements

- better model validation and hyperparameter tuning
- richer app feedback and explainability
- clearer documentation and usage instructions
- unit tests and reproducible deployment setup

## How to use

1. Run `python train_model.py` to create model artifacts.
2. Run `streamlit run app.py` to use the interactive app.
3. Run `python predict.py` for command-line prediction.

More project details and features will be added as development continues.
