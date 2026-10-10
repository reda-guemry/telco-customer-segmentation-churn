from sklearn.base import BaseEstimator, TransformerMixin


class BuildFeatures(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        
        X = self.add_monthly_charge_per_service(X)
        
        return X

    def add_monthly_charge_per_service(self, X):
        X = X.copy()

        X["MonthlyChargesPerService"] = (
            X["MonthlyCharges"]
            / X[
                [
                    "PhoneService",
                    "MultipleLines",
                    "OnlineSecurity",
                    "OnlineBackup",
                    "DeviceProtection",
                    "TechSupport",
                    "StreamingTV",
                    "StreamingMovies",
                ].sum(axis=1)
            ]
        )
        return X 
