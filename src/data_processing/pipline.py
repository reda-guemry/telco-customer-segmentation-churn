from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

import numpy as np
import pandas as pd

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
    # TotalCharges may contain blanks when read straight from the raw CSV; the
    # loader already coerces it to numeric (NaN for blanks). This keeps the
    # transform vectorised so it can be applied to a whole column by sklearn.
    values = pd.to_numeric(pd.Series(np.asarray(x).ravel()), errors="coerce")
    values = values.fillna(0.0).to_numpy(dtype=float)
    return np.log1p(values).reshape(-1, 1)


# TotalCharges is handled by its own (log) transformer above, so it must not be
# duplicated inside the generic numerical block.
NUMERICAL_FEATURES = [c for c in NUMERICAL_COLUMNS if c not in TOTAL_CHARGES_COLUMN]


def get_preprocessor() -> ColumnTransformer:

    preprocessor = ColumnTransformer(
        [
            (
                "binary",
                FunctionTransformer(binary_encoding, feature_names_out="one-to-one"),
                BINARY_COLUMNS,
            ),
            (
                "gender",
                FunctionTransformer(gender_encoding, feature_names_out="one-to-one"),
                GENDER_COLUMNS,
            ),
            (
                "total_charges",
                FunctionTransformer(
                    total_charges_cleaning, feature_names_out="one-to-one"
                ),
                TOTAL_CHARGES_COLUMN,
            ),
            (
                "numerical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERICAL_FEATURES,
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
