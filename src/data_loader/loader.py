import pandas as pd


from src import ROW_DATA_DIR
from src.data_loader import clean_types


def loader() -> pd.DataFrame:
    """
    Load the data from the specified directory.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    df = pd.read_csv(ROW_DATA_DIR)

    df = clean_types(df)

    return df
