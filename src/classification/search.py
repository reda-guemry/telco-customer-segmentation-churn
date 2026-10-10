"""MLflow-driven hyperparameter search for the churn classification models.

This module trains every pipeline returned by
``src.models_pip.final_models`` (preprocessor + KMeansTransformer + SMOTE +
classifier) over a randomized hyperparameter search and logs everything to
MLflow:

* the hyperparameters of every trial (``mlflow.log_params``);
* the cross-validation score and the test metrics of every trial
  (``mlflow.log_metrics``);
* a confusion-matrix and an ROC-curve figure for every trial
  (``mlflow.log_figure``);
* the best model of every algorithm and the overall best model
  (``mlflow.sklearn.log_model`` + a ``joblib`` artifact).

The run hierarchy mirrors the clustering search::

    Classification Search              (parent)
    ├── logistic_regression Search     (model)
    │   ├── logistic_regression_trial_1 (trial)
    │   └── ...
    ├── decision_tree Search
    └── ...

The best model of every algorithm is saved as ``best_<name>.pkl`` and the
overall best as ``best_model.pkl`` (the file consumed by the inference app).
"""


import os

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import (
    ParameterGrid,
    ParameterSampler,
    cross_val_score,
)

from src.classification.evaluation import (
    POSITIVE_LABEL,
    confusion_matrix_figure,
    evaluate_classification,
    roc_curve_figure,
)
from src.config import CLASSIFICATION_PARAM_GRIDS


def _normalize_n_iter(n_iter, model_name: str, default: int = 20) -> int:
    """Resolve the per-model trial budget from an ``int`` or a mapping."""
    if isinstance(n_iter, dict):
        return int(n_iter.get(model_name, default))
    return int(n_iter)


def _positive_index(estimator, positive_label: int = POSITIVE_LABEL) -> int:
    """Column of ``predict_proba`` that holds the positive (churn) class."""
    classes = list(getattr(estimator, "classes_", [0, 1]))
    if positive_label in classes:
        return classes.index(positive_label)
    return len(classes) - 1


def _numeric_metrics(metrics: dict) -> dict:
    """Keep only the numeric entries of a metrics dict (MLflow metrics only)."""
    return {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float, np.integer, np.floating))
    }


def _log_figures(
    y_true,
    y_pred,
    y_proba,
    title: str,
    log_plots: bool = True,
) -> None:
    """Log the confusion-matrix and ROC-curve figures of a trial."""
    if not log_plots:
        return

    cm_fig = confusion_matrix_figure(y_true, y_pred, title=f"{title} - Confusion Matrix")
    mlflow.log_figure(cm_fig, "confusion_matrix.png")
    plt.close(cm_fig)

    roc_fig = roc_curve_figure(y_true, y_proba, model_name=title)
    mlflow.log_figure(roc_fig, "roc_curve.png")
    plt.close(roc_fig)


def _log_and_save_model(
    estimator,
    models_dir: str,
    file_name: str,
    artifact_path: str = "model",
) -> None:
    """Save a fitted classification pipeline with joblib and log it to MLflow."""
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, file_name)
    joblib.dump(estimator, path)
    mlflow.log_artifact(path, artifact_path="joblib")

    try:
        mlflow.sklearn.log_model(
            estimator,
            name=artifact_path,
            serialization_format="cloudpickle",
        )
    except Exception as exc:  # pragma: no cover - depends on the sklearn version
        mlflow.set_tag("model_log_error", str(exc)[:250])


def run_classification_experiment(
    models: dict,
    X_train,
    y_train,
    X_test,
    y_test,
    experiment_name: str = "Churn Prediction",
    parent_run_name: str = "Classification Search",
    param_grids: dict | None = None,
    n_iter=None,
    scoring: str = "recall",
    cv: int = 5,
    random_state: int = 42,
    log_plots: bool = True,
    models_dir: str = "models",
) -> dict | None:
    """Run the full MLflow classification search.

    Parameters
    ----------
    models : dict
        Mapping ``model_name -> unfitted pipeline``, typically the result of
        ``src.models_pip.final_models(preprocessor)``. Pipelines are cloned for
        every trial so the search never mutates the inputs.
    X_train, y_train, X_test, y_test : array-like
        Raw customer features and 0/1 churn labels. The pipelines embed their
        own preprocessor, so raw (un-processed) rows are expected.
    experiment_name : str
        MLflow experiment name.
    parent_run_name : str
        Name of the parent MLflow run.
    param_grids : dict, optional
        Mapping ``model_name -> param_grid``. Defaults to
        :data:`src.config.CLASSIFICATION_PARAM_GRIDS`.
    n_iter : int or dict, optional
        Number of random combinations sampled per model. Pass a mapping to use a
        different budget per model. Defaults to ``20``.
    scoring : str
        scikit-learn scorer used for the 5-fold cross-validation and to select
        the best trial (defaults to ``recall``).
    cv : int
        Number of cross-validation folds.
    random_state : int
        Random state used for the parameter sampling.
    log_plots : bool
        Whether to log a confusion-matrix and an ROC-curve per trial.
    models_dir : str
        Local directory where the best models are saved with joblib.

    Returns
    -------
    dict or None
        Description of the overall best model, or ``None`` when no model could
        be trained.
    """
    if not models:
        raise ValueError("`models` must contain at least one classifier pipeline.")

    n_iter = 20 if n_iter is None else n_iter

    mlflow.set_experiment(experiment_name)

    best_overall = None
    summary_rows: list[dict] = []

    with mlflow.start_run(run_name=parent_run_name) as parent_run:
        mlflow.set_tag("run_type", "parent")
        mlflow.set_tag("task", "classification")
        mlflow.log_param("n_models", len(models))
        mlflow.log_param("cv_folds", cv)
        mlflow.log_param("scoring", scoring)
        mlflow.log_param("selection_metric", scoring)
        mlflow.log_param("models", ", ".join(models))

        for model_name, base_model in models.items():
            grid = param_grids.get(model_name, {})

            if grid:
                total_combinations = len(list(ParameterGrid(grid)))
                n_trials = max(1, min(_normalize_n_iter(n_iter, model_name, 20), total_combinations))
                combinations = list(
                    ParameterSampler(
                        grid,
                        n_iter=n_trials,
                        random_state=random_state,
                    )
                )
            else:
                combinations = [{}]

            with mlflow.start_run(
                run_name=f"{model_name} Search",
                nested=True,
            ) as model_run:
                mlflow.set_tag("run_type", "model")
                mlflow.set_tag("model", model_name)
                mlflow.set_tag("parent_run_id", parent_run.info.run_id)
                mlflow.log_param("n_trials", len(combinations))
                mlflow.log_param("total_combinations", total_combinations if grid else 1)

                best_model_metrics = None
                best_model_params = None
                best_model_estimator = None
                best_model_trial = None
                best_model_cv = -float("inf")

                for trial_number, params in enumerate(combinations, start=1):
                    with mlflow.start_run(
                        run_name=f"{model_name}_trial_{trial_number}",
                        nested=True,
                    ) as trial_run:
                        mlflow.set_tag("run_type", "trial")
                        mlflow.set_tag("model", model_name)
                        mlflow.set_tag("parent_run_id", model_run.info.run_id)
                        mlflow.log_param("trial_number", trial_number)
                        if params:
                            mlflow.log_params(params)

                        # Clone so each trial is independent from the base pipeline.
                        estimator = clone(base_model)
                        if params:
                            estimator.set_params(**params)

                        cv_scores = cross_val_score(
                            estimator,
                            X_train,
                            y_train,
                            cv=cv,
                            scoring=scoring,
                            n_jobs=-1,
                        )
                        cv_mean = float(cv_scores.mean())
                        cv_std = float(cv_scores.std())
                        mlflow.log_metric(f"cv_{scoring}_mean", cv_mean)
                        mlflow.log_metric(f"cv_{scoring}_std", cv_std)

                        # Fit on the complete training set and score the test set.
                        estimator.fit(X_train, y_train)
                        y_pred = estimator.predict(X_test)
                        y_proba = estimator.predict_proba(X_test)[:, _positive_index(estimator)]
                        metrics = evaluate_classification(y_test, y_pred, y_proba) 
                        mlflow.log_metrics(_numeric_metrics(metrics))

                        _log_figures(
                            y_test,
                            y_pred,
                            y_proba,
                            title=f"{model_name} - trial {trial_number}",
                            log_plots=log_plots,
                        )

                        summary_rows.append(
                            {
                                "model": model_name,
                                "trial_number": trial_number,
                                **params,
                                f"cv_{scoring}_mean": cv_mean,
                                f"cv_{scoring}_std": cv_std,
                                **metrics,
                            }
                        )

                        if cv_mean > best_model_cv:
                            best_model_cv = cv_mean
                            best_model_metrics = metrics.copy()
                            best_model_params = params.copy()
                            best_model_estimator = estimator
                            best_model_trial = trial_number
                            mlflow.set_tag("best_trial", "true")

                if best_model_metrics is None:
                    mlflow.set_tag("status", "no_valid_trial")
                    continue

                # Best model of this algorithm.
                mlflow.set_tag("best_trial_number", best_model_trial)
                mlflow.log_metric(f"best_cv_{scoring}", best_model_cv)
                mlflow.log_metrics(_numeric_metrics(best_model_metrics))
                if best_model_params:
                    mlflow.log_params(
                        {f"best_{key}": value for key, value in best_model_params.items()}
                    )

                _log_and_save_model(
                    best_model_estimator,
                    models_dir,
                    file_name=f"best_{model_name}.pkl",
                )

                if best_overall is None or best_model_cv > best_overall["cv_score"]:
                    best_overall = {
                        "model": model_name,
                        "trial_number": best_model_trial,
                        "params": best_model_params,
                        "metrics": best_model_metrics,
                        "cv_score": best_model_cv,
                        "estimator": best_model_estimator,
                    }

        if summary_rows:
            summary = pd.DataFrame(summary_rows)
            summary["is_best"] = False
            if best_overall is not None:
                is_best = (
                    (summary["model"] == best_overall["model"])
                    & (summary["trial_number"] == best_overall["trial_number"])
                )
                summary.loc[is_best, "is_best"] = True
            mlflow.log_text(
                summary.to_csv(index=False),
                "classification_trials_summary.csv",
            )

        if best_overall is None:
            mlflow.set_tag("status", "no_valid_trial")
            return None

        mlflow.set_tag("status", "completed")
        mlflow.set_tag("best_model", best_overall["model"])
        mlflow.set_tag("best_trial_number", best_overall["trial_number"])
        mlflow.log_param("best_model", best_overall["model"])
        mlflow.log_metric(f"best_cv_{scoring}", best_overall["cv_score"])
        mlflow.log_metrics(_numeric_metrics(best_overall["metrics"]))
        if best_overall["params"]:
            mlflow.log_params(
                {f"best_{key}": value for key, value in best_overall["params"].items()}
            )

        _log_and_save_model(
            best_overall["estimator"],
            models_dir,
            file_name="best_model.pkl",
            artifact_path="best_model",
        )

    return best_overall
