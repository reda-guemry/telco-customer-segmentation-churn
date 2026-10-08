

    
import pandas as pd


def clean_types(df, numerical_columns): 
    """
    Clean the data types of the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with cleaned data types.
    """
    for col in numerical_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df
