import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal
import pytest
import sys
import os
import sys


current_dir = os.getcwd()
data_dir = os.path.join(os.path.dirname(current_dir), 'data')
file_path = os.path.join(data_dir, 'sample_df.csv')

sample_df = pd.read_csv(file_path)

import os
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV
from src.eda import eda_dataframe


import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

def test_empty_string_count_and_drop():
    # Create a sample DataFrame
    data = sample_df
    df = pd.DataFrame(data)

    # Process the DataFrame
    processed_df = eda_dataframe(df.copy())

    # Check if columns with more than 50,000 empty strings are dropped
    assert 'col3' not in processed_df.columns
    # Check if other columns are still there
    assert 'col1' in processed_df.columns
    assert 'col2' in processed_df.columns

def test_predefined_columns_drop():
    # Create a sample DataFrame with predefined columns
    predefined_cols = ['echoBuffer', 'merchantCity', 'merchantZip', 'posOnPremises', 'recurringAuthInd', 'merchantState']
    data = {col: ['value']*100 for col in predefined_cols}
    df = pd.DataFrame(data)

    # Process the DataFrame
    processed_df = eda_dataframe(df.copy())

    for col in predefined_cols:
        assert col not in processed_df.columns

def test_inplace_modification():
    # Create a sample DataFrame
    data = df
    df = pd.DataFrame(data)

    # Store the original id of the DataFrame
    original_id = id(df)

    # Process the DataFrame
    eda_dataframe(df)

    # Check if the DataFrame is modified in-place
    assert id(df) == original_id

def test_no_column_drop_required():
    # Create a sample DataFrame without empty strings and predefined columns
    data = df
    df = pd.DataFrame(data)

    processed_df = eda_dataframe(df.copy())

    assert processed_df.equals(df)

