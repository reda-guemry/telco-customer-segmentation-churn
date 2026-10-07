from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline


from src.kmean import KMeansTransformer


def final_models(preprocessor):
    models = {
        "logistic_regression": Pipeline(
            [
                ("preprocessor", preprocessor),
                ("Kmeans", KMeansTransformer(n_clusters=3, random_state=42)), 
                ("smote", SMOTE(random_state=42)),
                ("classifier", LogisticRegression()),
            ]
        ),
        "decision_tree": Pipeline(
            [
                ("preprocessor", preprocessor),
                ("Kmeans", KMeansTransformer(n_clusters=3, random_state=42)), 
                ("smote", SMOTE(random_state=42)),
                ("classifier", DecisionTreeClassifier()),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("preprocessor", preprocessor),
                ("Kmeans", KMeansTransformer(n_clusters=3, random_state=42)), 
                ("smote", SMOTE(random_state=42)),
                ("classifier", RandomForestClassifier()),
            ]
        ),
        "SVC": Pipeline(
            [
                ("preprocessor", preprocessor),
                ("Kmeans", KMeansTransformer(n_clusters=3, random_state=42)), 
                ("smote", SMOTE(random_state=42)),
                ("classifier", SVC(probability=True)),
            ]
        ),
        "XGBoost": Pipeline(
            [
                ("preprocessor", preprocessor),
                ("Kmeans", KMeansTransformer(n_clusters=3, random_state=42)), 
                ("smote", SMOTE(random_state=42)),
                ("classifier", XGBClassifier()),
            ]
        ),
    }

    return models
