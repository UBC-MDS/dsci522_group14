import pandas as pd
import numpy as np

# Acknowledgement:
# The format of the function documentations are derived from the sample project provided. 
# Proper citation to the sample projct is included in `notebooks/fraud_detection.ipynb`. 
# https://github.com/ttimbers/demo-tests-ds-analysis-python/blob/main/src/count_classes.py

def count_unique_numbers(df, feature_list):
    """
    Tallies the count of distinct observations for each designated feature. 

    Outputs a new DataFrame containing two columns: one indicating the feature names
    and the other representing the count of unique observations for each respective feature.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame containing the data to analyze.
    feature_list : list
        The list of categorical features that will be analyzed. 

    Returns:
    --------
    pandas.DataFrame
        Dataframe with a list of categorical features and the unique number of observations per feature. 

    Examples:
    --------
    >>> import pandas as pd
    >>> data = pd.read_pickle('data/transactions.pkl.zip', compression="infer")
    >>> result = count_unique_numbers(data, ["merchantName", "acqCountry", "isFraud"])
    >>> print(result)

    """
      
    col_name = []
    counts = []
    for feature in feature_list:
        col_name.append(feature)
        counts.append(len(df[feature].unique().tolist()))

    return pd.DataFrame({"feature":col_name, 
                            "unique_entry_counts": counts}
                            ).sort_values(by="unique_entry_counts", ascending=False)


def generalize_categories(column, df):
    """
    Evaluates the occurrence frequency of each observation for categorical features. 
    Computes the frequency, ranging from 0 (no occurrences) to 1 (all observations are identical).

    Parameters:
    -----------
    column : str
        The name of the feature used to determine frequency of occurence.
    df : list
        The input DataFrame containing the data to analyze.

    Returns:
    --------
    pandas.DataFrame
        The original DataFrame now incorporates an extra column labeled 'frequency.' 
        This newly added column, specifically for the designated feature, provides details 
        about the occurrence frequency of the corresponding observations.

    Examples:
    --------
    >>> import pandas as pd
    >>> data = pd.read_pickle('data/transactions.pkl.zip', compression="infer")
    >>> result = generalize_categories("merchantName", data)
    >>> print(result)

    """

    count_df = pd.DataFrame(df[column].value_counts())
    count_df.columns = ["count"]
    count_df["frequency"] = (count_df["count"] / count_df["count"].sum())
    count_df = count_df.drop(["count"], axis=1)
    count_freq = count_df.to_dict()["frequency"]

    origin_df = df.copy()
    origin_df["frequency"] = origin_df[column].map(count_freq)
    organize_counts = lambda row: "Other" if row["frequency"] <= 0.01 else row[column]
    origin_df[column] = origin_df.apply(organize_counts, axis=1)
    origin_df[column] = origin_df[column].astype("str")
    
    return origin_df