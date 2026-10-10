"""Hyperparameter search spaces shared across the project."""

# Grid searched for every clustering algorithm. Keys are the exact names used to
# build the estimators (see ``src.clustering.search.CLUSTERERS``).
CLUSTERING_PARAM_GRIDS = {
    "KMeans": {
        "n_clusters": [2, 3, 4, 5, 6],
        "init": ["k-means++"],
        "n_init": [10, 20],
        "max_iter": [300],
        "random_state": [42],
    },
    "DBSCAN": {
        "eps": [1.4, 1.5, 1.8, 2.0, 2.2, 2.4, 2.5],
        "min_samples": [3, 5, 7, 10],
        "metric": ["euclidean"],
    },
    "AgglomerativeClustering": {
        "n_clusters": [2, 3, 4, 5, 6],
        "linkage": ["ward", "complete", "average", "single"],
        "metric": ["euclidean"],
    },
}

# Metrics for which a smaller value means a better model. Used when comparing
# trial results to select the best clustering run.
LOWER_IS_BETTER = ("davies_bouldin_score", "noise_percentage")


# Grid searched for every churn classification model. Keys match the model
# names returned by ``src.models_pip.final_models`` and the parameters use the
# ``classifier__`` prefix of the pipeline's last step.
CLASSIFICATION_PARAM_GRIDS = {
    "logistic_regression": {
        "classifier__C": [0.01, 0.1, 1, 10, 100],
        "classifier__solver": ["liblinear", "saga"],
        "classifier__max_iter": [400, 600, 800, 1000, 1500],
        "classifier__class_weight": [None, "balanced"],
    },
    "decision_tree": {
        "classifier__criterion": ["gini", "entropy"],
        "classifier__max_depth": [None, 5, 10, 15, 20],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__class_weight": [None, "balanced"],
    },
    "random_forest": {
        "classifier__n_estimators": [100, 200, 300, 400, 500],
        "classifier__max_depth": [None, 5, 10, 15, 20],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__class_weight": [None, "balanced"],
    },
    "SVC": {
        "classifier__C": [0.1, 1, 10],
        "classifier__kernel": ["rbf", "linear"],
        "classifier__gamma": ["scale", "auto"],
        "classifier__class_weight": [None, "balanced"],
    },
    "XGBoost": {
        "classifier__n_estimators": [100, 200, 300, 400, 500],
        "classifier__max_depth": [3, 5, 7, 9],
        "classifier__learning_rate": [0.01, 0.1, 0.2, 0.3],
        "classifier__subsample": [0.5, 0.7, 1.0],
        "classifier__colsample_bytree": [0.5, 0.7, 1.0],
        "classifier__scale_pos_weight": [1, 2, 3, 4, 5],
    },
}
