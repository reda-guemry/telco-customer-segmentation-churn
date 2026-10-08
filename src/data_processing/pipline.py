from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

import numpy as np

from src import (
    CATEGORICAL_COLUMNS,
    NUMERICAL_COLUMNS,
    BINARY_COLUMNS,
    GENDER_COLUMNS,
    TOTAL_CHARGES_COLUMN,
    loader,
)

def binary_encoding(x):
    return x.replace({"Yes": 1, "No": 0})


def gender_encoding(x):
    return x.replace({"Male": 1, "Female": 0})

def total_charges_cleaning(x):
    if isinstance(x, str) or np.isnan(x) :
        x = x.replace(" ", 0)

    return np.log1p(float(x))

def get_preprocessor() -> ColumnTransformer:

    preprocessor = ColumnTransformer(
        [
            ("binary", Pipeline([("encoding", binary_encoding)]), BINARY_COLUMNS),
            ("gender", Pipeline([("encoding", gender_encoding)]), GENDER_COLUMNS),
            ("total_charges", Pipeline([("cleaning", total_charges_cleaning)]), TOTAL_CHARGES_COLUMN),
            (
                "numerical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERICAL_COLUMNS,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoding",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                CATEGORICAL_COLUMNS,
            ),
        ],
        remainder="drop",
    )

    return preprocessor
