"""MLflow-driven hyperparameter search for clustering algorithms.

This module trains ``KMeans``, ``DBSCAN`` and ``AgglomerativeClustering`` over
their respective hyperparameter grids and logs everything to MLflow:

* the hyperparameters of every trial (``mlflow.log_params``);
* the evaluation metrics of every trial (``mlflow.log_metrics``);
* the clustering result (labels + cluster sizes) and a PCA figure
  (``mlflow.log_dict`` / ``mlflow.log_text`` / ``mlflow.log_figure``);
* the best model of every algorithm and the overall best model
  (``mlflow.sklearn.log_model`` + a ``joblib`` artifact).

The run hierarchy is::

    Clustering Search                 (parent)
    ├── KMeans Search                 (algorithm)
    │   ├── KMeans_trial_1            (trial)
    │   └── ...
    ├── DBSCAN Search
    └── AgglomerativeClustering Search
"""

from __future__ import annotations

import os

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline

from src.clustering.evaluation import (
    evaluate_clustering,
    labels_to_frame,
)
from src.config import CLUSTERING_PARAM_GRIDS, LOWER_IS_BETTER

# Registry mapping the algorithm name to its scikit-learn implementation.
CLUSTERERS = {
    "KMeans": KMeans,
    "DBSCAN": DBSCAN,
    "AgglomerativeClustering": AgglomerativeClustering,
}


def _build_estimator(algorithm: str, params: dict):
    """Instantiate a clustering estimator from its algorithm name and params."""
    return CLUSTERERS[algorithm](**params)


def _numeric_metrics(metrics: dict) -> dict:
    """Keep only the numeric entries of a metrics dict (MLflow metrics only)."""
    return {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float, np.integer, np.floating))
    }


def _is_better(candidate: dict | None, best: dict | None, metric: str) -> bool:
    """Tell whether ``candidate`` beats ``best`` according to ``metric``."""
    if candidate is None or metric not in candidate:
        return False
    if best is None or metric not in best:
        return True

    if metric in LOWER_IS_BETTER:
        return candidate[metric] < best[metric]
    return candidate[metric] > best[metric]


def _log_figure(X_pca, labels, title: str, artifact_file: str) -> None:
    """Log a PCA scatter plot coloured by cluster to MLflow."""
    X_pca = np.asarray(X_pca)
    labels = np.asarray(labels)

    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=labels,
        cmap="tab20",
        s=8,
        alpha=0.7,
    )
    ax.set_title(title)
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    fig.colorbar(scatter, ax=ax, label="cluster")
    fig.tight_layout()

    mlflow.log_figure(fig, artifact_file)
    plt.close(fig)


def _log_clustering_result(
    labels,
    index,
    X_pca,
    title: str,
    metrics: dict,
    log_plots: bool = True,
) -> None:
    """Log the clustering evaluation metrics, labels, sizes and PCA figure."""
    mlflow.log_dict(metrics, "evaluation_metrics.json")

    labels_frame = labels_to_frame(labels, index=index)
    mlflow.log_text(labels_frame.to_csv(), "clustering_results.csv")

    sizes = labels_frame["cluster"].value_counts().sort_index()
    sizes_frame = sizes.rename("n_samples").to_frame()
    sizes_frame["percentage"] = (sizes / sizes.sum() * 100).round(4)
    mlflow.log_text(sizes_frame.to_csv(), "cluster_sizes.csv")

    if log_plots:
        _log_figure(X_pca, labels, title, "clusters_pca.png")


def _wrap_model(estimator, preprocessor):
    """Wrap a fitted estimator with the fitted preprocessor when available."""
    if preprocessor is None:
        return estimator

    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("cluster", estimator),
        ]
    )


def _log_and_save_model(
    estimator,
    preprocessor,
    models_dir: str,
    file_name: str,
    artifact_path: str = "model",
) -> None:
    """Save a clustering model with joblib and log it to MLflow.

    Clustering estimators such as ``DBSCAN`` and ``AgglomerativeClustering`` do
    not implement ``predict``/``predict_proba``. MLflow can still serialize them,
    but to stay robust the joblib artifact is always logged and the
    ``mlflow.sklearn.log_model`` call is wrapped in a ``try/except``.
    """
    model = _wrap_model(estimator, preprocessor)

    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, file_name)
    joblib.dump(model, path)
    mlflow.log_artifact(path, artifact_path="joblib")

    try:
        mlflow.sklearn.log_model(
            model,
            name=artifact_path,
            serialization_format="cloudpickle",
        )
    except Exception as exc:  # pragma: no cover - depends on the sklearn version
        mlflow.set_tag("model_log_error", str(exc)[:250])


def run_clustering_experiment(
    X,
    df_index=None,
    preprocessor=None,
    experiment_name: str = "Customer Segmentation",
    parent_run_name: str = "Clustering Search",
    param_grids: dict | None = None,
    selection_metric: str = "silhouette_score",
    sample_size: int | None = None,
    random_state: int = 42,
    log_plots: bool = True,
    models_dir: str = "models",
) -> dict | None:
    """Run the full MLflow clustering search.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Preprocessed feature matrix (output of ``preprocessor.fit_transform``).
    df_index : array-like, optional
        Index used when logging the cluster labels (defaults to ``range(n)``).
    preprocessor : sklearn transformer, optional
        Fitted preprocessor. When provided it is bundled with the best models so
        that the logged models accept the raw customer DataFrame.
    experiment_name : str
        MLflow experiment name.
    parent_run_name : str
        Name of the parent MLflow run.
    param_grids : dict, optional
        Mapping ``algorithm -> param_grid``. Defaults to
        :data:`src.config.CLUSTERING_PARAM_GRIDS`.
    selection_metric : str
        Metric used to select the best trial (defaults to the silhouette score).
    sample_size : int, optional
        Sub-sample size used for the silhouette score.
    random_state : int
        Random state used for PCA and the silhouette sub-sampling.
    log_plots : bool
        Whether to log a PCA figure for every trial.
    models_dir : str
        Local directory where the best models are saved with joblib.

    Returns
    -------
    dict or None
        Description of the overall best model, or ``None`` if no valid trial
        produced the ``selection_metric``.
    """
    param_grids = param_grids or CLUSTERING_PARAM_GRIDS
    
    X = np.asarray(X)
    if df_index is None:
        df_index = np.arange(X.shape[0])

    # 2D embedding reused for every clustering figure.
    X_pca = PCA(n_components=2, random_state=random_state).fit_transform(X)

    mlflow.set_experiment(experiment_name)

    best_overall = None
    summary_rows: list[dict] = []

    with mlflow.start_run(run_name=parent_run_name) as parent_run:
        mlflow.set_tag("run_type", "parent")
        mlflow.set_tag("task", "clustering")
        mlflow.log_param("n_samples", X.shape[0])
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_param("selection_metric", selection_metric)
        mlflow.log_param("algorithms", ", ".join(param_grids))

        for algorithm, grid in param_grids.items():
            combinations = list(ParameterGrid(grid))
            mlflow.log_param(f"{algorithm}_n_trials", len(combinations))

            with mlflow.start_run(
                run_name=f"{algorithm} Search",
                nested=True,
            ) as algorithm_run:
                mlflow.set_tag("run_type", "algorithm")
                mlflow.set_tag("algorithm", algorithm)
                mlflow.set_tag("parent_run_id", parent_run.info.run_id)

                best_algo_metrics = None
                best_algo_params = None
                best_algo_estimator = None
                best_algo_labels = None
                best_algo_trial = None

                for trial_number, params in enumerate(combinations, start=1):
                    with mlflow.start_run(
                        run_name=f"{algorithm}_trial_{trial_number}",
                        nested=True,
                    ) as trial_run:
                        mlflow.set_tag("run_type", "trial")
                        mlflow.set_tag("algorithm", algorithm)
                        mlflow.set_tag("parent_run_id", algorithm_run.info.run_id)
                        mlflow.log_param("trial_number", trial_number)
                        mlflow.log_params(params)

                        estimator = _build_estimator(algorithm, params)
                        labels = estimator.fit_predict(X)

                        metrics = evaluate_clustering(
                            X,
                            labels,
                            algorithm,
                            sample_size=sample_size if sample_size is not None else None,
                            random_state=random_state,
                        )
                        mlflow.log_metrics(_numeric_metrics(metrics))

                        _log_clustering_result(
                            labels,
                            df_index,
                            X_pca,
                            title=f"{algorithm} - trial {trial_number}",
                            metrics=metrics,
                            log_plots=log_plots,
                        )

                        summary_rows.append(
                            {
                                "algorithm": algorithm,
                                "trial_number": trial_number,
                                **params,
                                **metrics,
                            }
                        )

                        if _is_better(metrics, best_algo_metrics, selection_metric):
                            best_algo_metrics = metrics.copy()
                            best_algo_params = params.copy()
                            best_algo_estimator = estimator
                            best_algo_labels = labels
                            best_algo_trial = trial_number
                            mlflow.set_tag("best_trial", "true")

                if best_algo_metrics is None:
                    mlflow.set_tag("status", "no_valid_trial")
                    continue

                # Best model of this algorithm.
                mlflow.set_tag("best_trial_number", best_algo_trial)
                mlflow.log_params(
                    {f"best_{key}": value for key, value in best_algo_params.items()}
                )
                mlflow.log_metrics(_numeric_metrics(best_algo_metrics))

                _log_clustering_result(
                    best_algo_labels,
                    df_index,
                    X_pca,
                    title=f"Best {algorithm}",
                    metrics=best_algo_metrics,
                    log_plots=log_plots,
                )

                _log_and_save_model(
                    best_algo_estimator,
                    preprocessor,
                    models_dir,
                    file_name=f"best_{algorithm}.pkl",
                )

                if _is_better(best_algo_metrics, best_overall, selection_metric):
                    best_overall = {
                        "algorithm": algorithm,
                        "trial_number": best_algo_trial,
                        "params": best_algo_params,
                        "metrics": best_algo_metrics,
                        "estimator": best_algo_estimator,
                        "labels": best_algo_labels,
                    }

        if summary_rows:
            summary = pd.DataFrame(summary_rows)
            summary["is_best"] = False
            if best_overall is not None:
                is_best = (
                    (summary["algorithm"] == best_overall["algorithm"])
                    & (summary["trial_number"] == best_overall["trial_number"])
                )
                summary.loc[is_best, "is_best"] = True
            mlflow.log_text(summary.to_csv(index=False), "clustering_trials_summary.csv")

        if best_overall is None:
            mlflow.set_tag("status", "no_valid_trial")
            return None

        mlflow.set_tag("status", "completed")
        mlflow.set_tag("best_algorithm", best_overall["algorithm"])
        mlflow.set_tag("best_trial_number", best_overall["trial_number"])
        mlflow.log_param("best_algorithm", best_overall["algorithm"])
        mlflow.log_params(
            {f"best_{key}": value for key, value in best_overall["params"].items()}
        )
        mlflow.log_metrics(_numeric_metrics(best_overall["metrics"]))

        _log_clustering_result(
            best_overall["labels"],
            df_index,
            X_pca,
            title=f"Overall best - {best_overall['algorithm']}",
            metrics=best_overall["metrics"],
            log_plots=log_plots,
        )

        _log_and_save_model(
            best_overall["estimator"],
            preprocessor,
            models_dir,
            file_name="best_clustering_model.pkl",
            artifact_path="best_model",
        )

    return best_overall
