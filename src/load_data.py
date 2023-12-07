import pandas as pd
import os


def load_data(file_path):
    """
    Loads the data from the data folder.

    Parameters:
    -----------
    filepath : string
        The input filepath to the data folder.

    Returns:
    --------
    pandas.DataFrame
        The DataFrame containing the data from the data folder.

    Examples:
    --------
    >>> import pandas as pd
    >>> df = load_data('data/transactions.pkl.zip')
    >>> print(df)
    """
    if not os.path.exists(file_path):
        return None
    return pd.read_pickle(file_path, compression="infer")