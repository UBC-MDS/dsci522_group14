import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessing import count_unique_numbers, generalize_categories

# Import sample dataset
sample_df = pd.read_csv("data/sample_df.csv")
feature_list = ["merchantName", "acqCountry", "merchantCity", "isFraud"]

# Observed output for `count_unique_numbers`
subset_data = pd.DataFrame({
    "merchantName": len(sample_df["merchantName"].unique().tolist()),
    "acqCountry": len(sample_df["acqCountry"].unique().tolist()),
    "merchantCity": len(sample_df["merchantCity"].unique().tolist()),
    "isFraud": len(sample_df["isFraud"].unique().tolist())
}, index=["unique_entry_counts"]).T.reset_index()
subset_data.columns = ["feature", "unique_entry_counts"]
subset_data = subset_data.sort_values(by="unique_entry_counts", ascending=False)

mock_data = pd.DataFrame({
    "feature1": [1, 1, 1, 3, 4, 5],
    "feature2": [1, 2, 3, 4, 5, 6],
    "feature3": ["yes", "yes", "no", "yes", "neutral", "yes"]
})


# Test if the second column is ordered in a descending order for `count_unique_numbers`
def test_correct_number_count():
    expected = count_unique_numbers(sample_df, feature_list)
    assert expected['unique_entry_counts'].is_monotonic_decreasing


# Test for correct return type for `count_unique_numbers`
def test_correct_return_type():
    expected = count_unique_numbers(sample_df, feature_list)
    assert isinstance(expected, pd.DataFrame), "Returning incorrect data type; must be a Pandas dataframe."


# Test if the observed and expected data frames are equal for `count_unique_numbers`
def test_equal_dataframe():
    expected = count_unique_numbers(sample_df, feature_list)
    pd.testing.assert_frame_equal(expected, subset_data)


# Test if the frequency column is created in returned output for `generealize_categories`
def test_correct_frequency_column():
    expected1 = generalize_categories("feature1", mock_data)
    assert "frequency" in expected1.columns.tolist()


# Test for correct number of returned rows for `generealize_categories`
def test_number_of_rows():
    expected1 = generalize_categories("feature1", mock_data)
    assert expected1.shape[0] == mock_data.shape[0]


# Test if correct number of frequency per observation is computed for `generealize_categories`
def test_correct_frequency_column():
    expected1 = generalize_categories("feature1", mock_data)
    assert expected1["frequency"].tolist() == [0.5, 0.5, 0.5, 1 / 6, 1 / 6, 1 / 6]
