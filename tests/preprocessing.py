import pandas as pd
import pytest
import sys
import os

import sys
sys.path.append('../src')

from src.preprocessing import count_unique_numbers, generalize_categories

# Import sample dataset
sample_df = pd.read_csv("sample_df.csv")
feature_list = ["", "", ""]

def test_correct_number_count():
    expected = count_unique_numbers(sample_df, feature_list)

    observed = pd.DataFrame({
        ""
    })

# Test for correct return type
def test_count_classes_returns_dataframe():
    result = count_classes(two_classes_3_obs, 'class_labels')
    assert isinstance(result, pd.DataFrame), "count_classes` should return a pandas data frame"
