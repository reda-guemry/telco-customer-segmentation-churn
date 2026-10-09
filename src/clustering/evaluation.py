"""Evaluation utilities for unsupervised clustering experiments.

The helpers in this module are intentionally free of any MLflow dependency so
they can be reused from notebooks, scripts or tests.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


def evaluate_clustering(
    X,
    labels,
    model_name,
    mask=None,
    sample_size: int | None = None ,
    random_state: int = 42,
) -> dict:
    """Compute the internal evaluation metrics of a clustering.

    Noise points (``label == -1``, produced by DBSCAN) do not belong to any
    cluster, so they are removed before computing the metrics. This mirrors the
    logic used in ``notebooks/03_clustering.ipynb``.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Preprocessed feature matrix used to fit the clustering.
    labels : array-like of shape (n_samples,)
        Cluster labels returned by the fitted model.
    model_name : str
        Name of the clustering algorithm (stored in the returned dict).
    mask : array-like of bool, optional
        Boolean mask selecting the samples used to compute the metrics.
        Defaults to ``labels != -1``.
    sample_size : int, optional
        Number of samples used to estimate the (expensive) silhouette score.
        Set to ``None`` to use every sample.
    random_state : int
        Random state used for the silhouette sub-sampling.

    Returns
    -------
    dict
        ``model``, ``n_clusters``, ``noise_percentage`` and, when at least two
        clusters remain, ``silhouette_score``, ``davies_bouldin_score`` and
        ``calinski_harabasz_score``. All values are plain Python types so the
        result is JSON serializable (required by ``mlflow.log_dict``).
    """
    X = np.asarray(X)
    labels = np.asarray(labels)

    if mask is None:
        mask = labels != -1
    mask = np.asarray(mask)

    X_scored = X[mask]
    labels_scored = labels[mask]

    n_clusters = int(len(np.unique(labels_scored)))
    noise_pct = float((labels == -1).mean() * 100.0) if labels.size else 0.0

    metrics = {
        "model": str(model_name),
        "n_clusters": n_clusters,
        "noise_percentage": round(noise_pct, 4),
    }

    if n_clusters > 1:
        n_samples = X_scored.shape[0]
        sil_kwargs = {}
        if sample_size is not None and n_samples > sample_size:
            sil_kwargs = {
                "sample_size": max(sample_size, n_clusters + 1),
                "random_state": random_state,
            }

        metrics["silhouette_score"] = float(
            silhouette_score(X_scored, labels_scored, **sil_kwargs)
        )
        metrics["davies_bouldin_score"] = float(
            davies_bouldin_score(X_scored, labels_scored)
        )
        metrics["calinski_harabasz_score"] = float(
            calinski_harabasz_score(X_scored, labels_scored)
        )

    return metrics


def labels_to_frame(labels, index=None, column: str = "cluster") -> pd.DataFrame:
    """Return the clustering labels as a one-column :class:`pandas.DataFrame`."""
    labels = np.asarray(labels)
    if index is None:
        index = np.arange(labels.shape[0])

    return pd.DataFrame({column: labels}, index=index)

