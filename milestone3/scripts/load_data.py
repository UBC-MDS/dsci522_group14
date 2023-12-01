import pandas as pd
import os


def load_data():
    """
    Loads the data from the data folder.

    Returns:
    --------
    pandas.DataFrame
        The DataFrame containing the data from the data folder.

    Examples:
    --------
    >>> import pandas as pd
    >>> df = load_data()
    >>> print(df)
    """
    if not os.path.exists('../data/transactions.pkl.zip'):
        return None
    return pd.read_pickle('../data/transactions.pkl.zip', compression="infer")