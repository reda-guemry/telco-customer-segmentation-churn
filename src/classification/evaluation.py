"""Evaluation utilities for supervised churn classification experiments.

The helpers in this module are intentionally free of any MLflow dependency so
they can be reused from notebooks, scripts or tests. They mirror the metrics
used in ``notebooks/04_classification_churn.ipynb`` (accuracy, precision,
recall, f1-score and roc-auc on the churn class).
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

#: Positive (churn) class label.
POSITIVE_LABEL = 1


def evaluate_classification(
    y_true,
    y_pred,
    y_proba,
    positive_label: int = POSITIVE_LABEL,
) -> dict:
    """Compute the supervised metrics of a churn classifier.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth binary labels.
    y_pred : array-like of shape (n_samples,)
        Predicted binary labels.
    y_proba : array-like of shape (n_samples,)
        Probability of the positive (churn) class.
    positive_label : int
        Label of the churn class (defaults to ``1``).

    Returns
    -------
    dict
        ``accuracy``, ``precision``, ``recall``, ``f1_score`` and ``roc_auc``
        as plain Python floats so the result is JSON serializable (required by
        ``mlflow.log_dict``).
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                pos_label=positive_label,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_pred,
                pos_label=positive_label,
            )
        ),
        "f1_score": float(
            f1_score(
                y_true,
                y_pred,
                pos_label=positive_label,
            )
        ),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def confusion_matrix_figure(
    y_true,
    y_pred,
    title: str = "Confusion Matrix",
    normalize: bool = True,
):
    """Return a confusion matrix figure (row-normalized percentages by default)."""
    matrix = confusion_matrix(y_true, y_pred)
    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(
            matrix,
            row_sums,
            out=np.zeros_like(matrix, dtype=float),
            where=row_sums != 0,
        ) * 100.0

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["No churn", "Churn"],
    ).plot(ax=ax, values_format=".1f" if normalize else "d", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def roc_curve_figure(
    y_true,
    y_proba,
    model_name: str = "model",
):
    """Return an ROC curve figure annotated with the AUC."""
    auc = roc_auc_score(y_true, y_proba)
    fpr, tpr, _ = roc_curve(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title(f"ROC Curve - {model_name}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig
