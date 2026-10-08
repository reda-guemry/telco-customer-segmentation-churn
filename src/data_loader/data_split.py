import pandas as pd
from sklearn.model_selection import train_test_split


def data_split(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> tuple:

    X = df.drop(columns=["Churn"])
    Y = df["Churn"].map({"No": 0, "Yes": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        Y,
        test_size=test_size,
        random_state=random_state,
        stratify=Y, 
    )

    return X_train, X_test, y_train, y_test
