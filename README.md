# telco-customer-segmentation-churn

Customer segmentation and churn prediction on the Telco customer dataset.
Clustering models (KMeans, DBSCAN, AgglomerativeClustering) are trained and
tracked with **MLflow**, and a **Streamlit** app serves churn predictions.
Everything can run locally or with **Docker**.

## The Streamlit app

`app/streamlit_app.py` is a prediction interface: fill in a customer's
characteristics and it displays

- their **segment / cluster** and the segment profile;
- the **churn prediction** (Yes / No);
- the **churn probability**.

Model loading is isolated in `src/inference.py` (`load_predictor`) and reads
**only from the `models/` directory** — no MLflow at serving time. It uses the
output of the two training jobs:

- `models/best_model.pkl` — churn classification (`python -m src.train`);
- `models/best_clustering_model.pkl` (or `models/best_KMeans.pkl` when the
  overall best clustering model cannot `predict`, e.g. DBSCAN) — segments
  (`python -m src.train_clustering`).

If a clustering model is missing, the fitted KMeans step of the classification
pipeline is used as a fallback.

## Requirements

- Python `>= 3.14`
- [uv](https://docs.astral.sh/uv/) (local usage)
- Docker + Docker Compose (containerized usage)

## Local usage

```bash
# Install dependencies
uv sync

# Train the two jobs — they save the models the Streamlit app reads
python -m src.train_clustering   # -> models/best_clustering_model.pkl, models/best_KMeans.pkl
python -m src.train              # -> models/best_model.pkl

# Run the prediction app (loads the joblib files from models/)
streamlit run app/streamlit_app.py

# Optional: open the MLflow UI (tracking store: ./mlflow.db)
mlflow ui
```

The Streamlit app does not need MLflow; MLflow is only used by the training jobs
for tracking. The tracking backend can be overridden with the
`MLFLOW_TRACKING_URI` environment variable (see `.env.exemple`).

### MLflow experiment layout

`python -m src.train_clustering` creates the **Customer Segmentation**
experiment with the following hierarchy:

```
Clustering Search                 (parent: overall best model)
├── KMeans Search
│   ├── KMeans_trial_1 ... N       (every hyperparameter combination)
├── DBSCAN Search
└── AgglomerativeClustering Search
```

For every trial it logs the hyperparameters, the evaluation metrics
(`silhouette_score`, `davies_bouldin_score`, `calinski_harabasz_score`,
`n_clusters`, `noise_percentage`), the cluster labels/sizes and a PCA figure.
The best model of each algorithm and the overall best model are saved with
joblib in `models/` and logged as MLflow models.

`python -m src.train` creates the **Churn Prediction** experiment with the same
nested structure:

```
Classification Search              (parent: overall best model)
├── logistic_regression Search
│   ├── logistic_regression_trial_1 ... N
├── decision_tree Search
├── random_forest Search
├── SVC Search
└── XGBoost Search
```

Each candidate is a full pipeline (preprocessor + KMeans + SMOTE + classifier)
tuned with a randomized hyperparameter search (`CLASSIFICATION_PARAM_GRIDS` in
`src/config/hyperparameters.py`). The best trial of every model is selected by
the mean 5-fold cross-validation `recall` on the churn class; for every trial it
logs the hyperparameters, `cv_recall_mean`/`cv_recall_std` and the test metrics
(`accuracy`, `precision`, `recall`, `f1_score`, `roc_auc`), plus a confusion
matrix and an ROC curve. The best model of each algorithm is saved as
`models/best_<model>.pkl` and the overall best as `models/best_model.pkl`.

## Docker usage

The `docker-compose.yml` builds a single image and runs it as several services:

| Service | URL | Description |
|---|---|---|
| `mlflow` | http://localhost:5000 | MLflow tracking server (only for the training jobs) |
| `streamlit` | http://localhost:8501 | Churn & segmentation prediction app (no MLflow dependency) |
| `train-clustering` | – | Clustering search, writes `models/` (profile `train`) |
| `train-churn` | – | Churn search, writes `models/` (profile `train`) |

```bash
# 1. Train the two jobs (they write the joblib files into ./models)
docker compose --profile train run --rm train-clustering
docker compose --profile train run --rm train-churn

# 2. Start the prediction app (reads ./models, no MLflow)
docker compose up -d streamlit

# (optional) also start the MLflow server to browse the training runs
docker compose up -d mlflow

# Stop everything
docker compose down
```

The `streamlit` service mounts the host `./models` directory and does not
depend on `mlflow`. MLflow data persists in the `mlflow-data` named volume.

## Project structure

```
app/streamlit_app.py        Streamlit prediction interface
src/inference.py            Model loading + prediction (segment/churn)
src/clustering/             Clustering training + MLflow logging
src/classification/         Classification training + MLflow logging
src/train_clustering.py     Clustering training entry point
src/train.py                Churn training entry point
src/data_processing/        Preprocessing pipeline
Dockerfile                  Image used by all services
docker-compose.yml          MLflow + Streamlit (+ optional training)
```
