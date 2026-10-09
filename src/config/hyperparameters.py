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
