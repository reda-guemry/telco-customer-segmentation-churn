import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

import joblib

from sklearn.model_selection import (
    train_test_split,
    ParameterSampler,
    cross_val_score,
)
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from src.models_pip import final_models
from src.data_loader import loader, data_split
from src.config import TARGET
from src.data_processing import get_preprocessor

# Load data
    
df = loader()

preprocessor = get_preprocessor()
models = final_models(preprocessor)
x_train, x_test, y_train, y_test =  data_split(df, test_size=0.2, random_state=42)

# Model

model = models["logistic_regression"]

# Hyperparameters

param_grid = {
    "classifier__solver": ["saga"],
    "classifier__penalty": ["elasticnet"],
    "classifier__C": [0.01, 0.1, 1, 10],
    "classifier__l1_ratio": [1, 0.5, 0],
    "classifier__max_iter": [1000, 1500, 2000, 2500, 3000],
    "classifier__class_weight": [None, "balanced"],
}


# Generate 50 random combinations
param_samples = list(
    ParameterSampler(
        param_grid,
        n_iter=50,
        random_state=42,
    )
)

# MLflow Experiment

mlflow.set_experiment("Churn Prediction")


with mlflow.start_run(run_name="Logistic Regression Search") as parent_run:

    # Parent information
    mlflow.set_tag("run_type", "parent")
    mlflow.set_tag("model_type", "logistic_regression")

    mlflow.log_param("n_trials", 50)
    mlflow.log_param("cv_folds", 5)
    mlflow.log_param("scoring", "recall")

    best_score = -float("inf")
    best_result = None
    best_model = None
    best_params = None

    # Hyperparameter Search

    for trial_number, params in enumerate(param_samples, start=1):

        print(f"Trial {trial_number}/50")
        print(params)

        # Child Run

        with mlflow.start_run(
            run_name=f"trial_{trial_number}",
            nested=True,
        ) as child_run:

            mlflow.set_tag("run_type", "child")
            mlflow.set_tag(
                "parent_run_id",
                parent_run.info.run_id,
            )

            mlflow.log_param("trial_number", trial_number)

            # Log hyperparameters
            mlflow.log_params(params)

            # Create model

            trial_model = model.set_params(**params)

            # Cross Validation

            cv_scores = cross_val_score(
                trial_model,
                x_train,
                y_train,
                cv=5,
                scoring="recall",
                n_jobs=-1,
            )

            mean_recall = cv_scores.mean()
            std_recall = cv_scores.std()

            mlflow.log_metric(
                "cv_recall_mean",
                mean_recall,
            )

            mlflow.log_metric(
                "cv_recall_std",
                std_recall,
            )

            # Fit on complete training set

            trial_model.fit(
                x_train,
                y_train,
            )

            # Test evaluation

            prediction = trial_model.predict(x_test)

            probabilities = trial_model.predict_proba(x_test)[:, 1]

            result = {
                "accuracy": accuracy_score(
                    y_test,
                    prediction,
                ),
                "precision": precision_score(
                    y_test,
                    prediction,
                ),
                "recall": recall_score(
                    y_test,
                    prediction,
                ),
                "f1": f1_score(
                    y_test,
                    prediction,
                ),
                "roc_auc": roc_auc_score(
                    y_test,
                    probabilities,
                ),
            }

            mlflow.log_metrics(result)

            # Save best model

            if mean_recall > best_score :

                best_score = mean_recall
                best_result = result.copy()
                best_model = trial_model
                best_params = params.copy()

                mlflow.set_tag(
                    "best_trial",
                    "true",
                )

    # Log best model in Parent

    mlflow.log_metrics(best_result)

    mlflow.log_params({f"best_{key}": value for key, value in best_params.items()})

    # Final Confusion Matrix

    final_prediction = best_model.predict(x_test)

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        final_prediction,
    )

    plt.title("Confusion Matrix - Best Model")
    plt.tight_layout()

    plt.savefig("confusion_matrix.png")

    plt.close()

    mlflow.log_artifact(
        "confusion_matrix.png",
        artifact_path="figures",
    )

    # Log final model

    mlflow.sklearn.log_model(
        best_model,
        name="model",
        serialization_format="cloudpickle",
    )

    joblib.dump(
        best_model,
        "models/best_model.pkl",
    )


