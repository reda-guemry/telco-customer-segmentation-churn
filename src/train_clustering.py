"""Train and track the clustering models (KMeans, DBSCAN, Agglomerative).

Run from the project root::

    python -m src.train_clustering

The hyperparameters of every candidate (see
``src.config.CLUSTERING_PARAM_GRIDS``), the clustering result and the
evaluation metrics are logged to MLflow. The best model of every algorithm and
the overall best model are saved in ``models/`` and logged as MLflow models.
"""

from __future__ import annotations

import os

import mlflow
import pandas as pd

from src.config import CLUSTERING_PARAM_GRIDS
from src.clustering import run_clustering_experiment
from src.data_loader import loader
from src.data_processing import get_preprocessor


def main() -> None:
    # Allow overriding the tracking backend (defaults to the local mlflow.db).
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    df = loader()
    preprocessor = get_preprocessor()

    # Clustering is unsupervised: fit the preprocessor on the full dataset and
    # reuse the transformed matrix for every candidate.
    X = preprocessor.fit_transform(df)
    X = pd.DataFrame(
        X,
        columns=preprocessor.get_feature_names_out(),
        index=df.index,
    )

    run_clustering_experiment(
        X,
        df_index=df.index,
        preprocessor=preprocessor,
        experiment_name="Customer Segmentation",
        parent_run_name="Clustering Search",
        param_grids=CLUSTERING_PARAM_GRIDS,
        selection_metric="silhouette_score",
        sample_size=3500,
        random_state=42,
        log_plots=True,
        models_dir="models",
    )


if __name__ == "__main__":
    main()
