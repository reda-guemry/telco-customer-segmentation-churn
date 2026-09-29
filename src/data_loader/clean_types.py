

from src import NUMERICAL_COLUMNS

import pandas as pd


def clean_types(df): 
    """
    Clean the data types of the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with cleaned data types.
    """
    for col in NUMERICAL_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df
