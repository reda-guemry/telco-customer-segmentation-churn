from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
import numpy as np

class KMeansTransformer(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        n_clusters=3,
        random_state=42,
        n_init=10,
    ):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init

    def fit(self, X, y=None) : 
        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=self.n_init,
        )

        self.kmeans_.fit(X) 

        return self 

    def transform(self, X) : 
        clusters = self.kmeans_.predict(X) 
        return np.column_stack((X, clusters))