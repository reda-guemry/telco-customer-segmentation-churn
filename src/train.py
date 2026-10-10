"""Train and track all churn classification models.

Run from the project root::

    python -m src.train

Every pipeline returned by ``src.models_pip.final_models`` (LogisticRegression,
DecisionTree, RandomForest, SVC, XGBoost -- each wrapped with the preprocessor,
the KMeans transformer and SMOTE) is tuned with a randomized hyperparameter
search (see ``src.config.CLASSIFICATION_PARAM_GRIDS``). The cross-validation
score, the test metrics and figures of every trial are logged to MLflow using
the same nested-run structure as the clustering job.

The best model of every algorithm is saved as ``models/best_<name>.pkl`` and the
overall best as ``models/best_model.pkl`` (the file consumed by the inference
app). The best model is selected by the mean cross-validation ``recall`` on the
churn class.
"""


import os

import mlflow

from src.classification import run_classification_experiment
from src.config import CLASSIFICATION_PARAM_GRIDS
from src.data_loader import data_split, loader
from src.data_processing import get_preprocessor
from src.models_pip import final_models

# Random-search budget per model. The full grids are large (up to ~3,600
# combinations for XGBoost), so each model samples a capped number of trials.
N_ITER = {
    "logistic_regression": 25,
    "decision_tree": 20,
    "random_forest": 20,
    "SVC": 15,
    "XGBoost": 20,
}


def main() -> None:
    # Allow overriding the tracking backend (defaults to the local mlflow.db).
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    df = loader()
    preprocessor = get_preprocessor()
    models = final_models(preprocessor)

    X_train, X_test, y_train, y_test = data_split(
        df,
        test_size=0.2,
        random_state=42,
    )

    run_classification_experiment(
        models,
        X_train,
        y_train,
        X_test,
        y_test,
        experiment_name="Churn Prediction",
        parent_run_name="Classification Search",
        param_grids=CLASSIFICATION_PARAM_GRIDS,
        n_iter=N_ITER,
        scoring="recall",
        cv=5,
        random_state=42,
        log_plots=True,
        models_dir="models",
    )


if __name__ == "__main__":
    main()
