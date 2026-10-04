from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

from src import (
    CATEGORICAL_COLUMNS,
    NUMERICAL_COLUMNS,
    BINARY_COLUMNS,
    GENDER_COLUMNS,
    loader,
)

binary_encoding = FunctionTransformer(
    lambda x: x.replace({"Yes": 1, "No": 0}), feature_names_out="one-to-one"
)
gender_encoding = FunctionTransformer(
    lambda x: x.replace({"Male": 1, "Female": 0}), feature_names_out="one-to-one"
)


def get_preprocessor() -> ColumnTransformer:

    preprocessor = ColumnTransformer(
        [
            (
                "numerical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="mean")),
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
            ("binary", Pipeline([("encoding", binary_encoding)]), BINARY_COLUMNS),
            ("gender", Pipeline([("encoding", gender_encoding)]), GENDER_COLUMNS),
        ],
        remainder="drop",
    )

    return preprocessor
