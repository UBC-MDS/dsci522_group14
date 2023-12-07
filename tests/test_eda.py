import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal
import pytest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.eda import eda_process

sample_df = pd.read_csv('data/sample_df.csv')


def test_empty_string_count_and_drop():
    # Create a sample DataFrame
    df = pd.DataFrame(columns=['col1', 'col2', 'col3'], data=[['keep', 'keep', '']] * 100000)

    # Process the DataFrame
    processed_df = eda_process(df)

    # Check if other columns are still there
    assert 'col1' in processed_df.columns
    assert 'col2' in processed_df.columns
    # Check if columns with more than 50,000 empty strings are dropped
    assert 'col3' not in processed_df.columns


def test_predefined_columns_drop():
    # Create a sample DataFrame with predefined columns
    predefined_cols = ['echoBuffer', 'merchantCity', 'merchantZip', 'posOnPremises', 'recurringAuthInd',
                       'merchantState']

    df = pd.DataFrame(sample_df)

    # Process the DataFrame
    processed_df = eda_process(df.copy())

    for col in predefined_cols:
        assert col not in processed_df.columns


def test_inplace_modification():
    # Create a sample DataFrame
    df = pd.DataFrame(sample_df)

    # Store the original id of the DataFrame
    original_id = id(df)

    # Process the DataFrame
    eda_process(df)

    # Check if the DataFrame is modified in-place
    assert id(df) == original_id


def test_no_column_drop_required():
    # Create a sample DataFrame without empty strings and predefined columns
    df = pd.DataFrame(columns=['col1', 'col2', 'col3'], data=[['keep', 'keep', 'keep']] * 100000)

    processed_df = eda_process(df.copy())

    assert processed_df.equals(df)
