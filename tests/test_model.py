import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal
import pytest
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model import run_model_with_random_search
# Assuming the `run_model_with_random_search` function is implemented 
# Import sample dataset
sample_df = pd.read_csv("data/sample_df.csv")
train_df, test_df = train_test_split(sample_df, test_size=0.2, random_state=522)
X_train, y_train = train_df.drop(["isFraud"], axis=1), train_df["isFraud"]
X_test, y_test = test_df.drop(["isFraud"], axis=1), test_df["isFraud"]

numerical_features=['availableMoney', 'transactionAmount', 'currentBalance']
categorical_features=['accountNumber', 'acqCountry', 'merchantCountryCode', 'posEntryMode', 'posConditionCode', 'merchantCategoryCode', 'transactionType', 'cardPresent', 'expirationDateKeyInMatch', 'merchantCity', 'merchantState', 'merchantZip', ]
credit_feature = ["creditLimit"]
drop_features=["customerId",'transactionDateTime', "merchantName","accountOpenDate","cardCVV","enteredCVV",'currentExpDate', 'cardLast4Digits', 'echoBuffer', 'dateOfLastAddressChange', 'posOnPremises', 'recurringAuthInd',]
# Tests for input data
def test_run_model_with_random_search_input_data():
    # Check if the input data is a DataFrame
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_test, pd.Series)

    # Check if the shapes of input data are as expected
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
    assert X_train.shape[1] == sample_df.shape[1]-1
    assert X_test.shape[1] == sample_df.shape[1]-1
    assert X_train.shape[1] == X_test.shape[1]
    
    # Add checks for specific column names, if needed
    assert "isFraud" == y_train.name
    assert "isFraud" == y_test.name


# Tests for run_model_with_random_search function
def test_run_model_with_random_search_logreg():
    param_dist_logreg = {"logisticregression__C": 10.0 ** np.arange(-2, 3),
                         "logisticregression__solver": ["newton-cholesky", "lbfgs"]}
    logreg_search = run_model_with_random_search(
        X_train, y_train, numerical_features, credit_feature, categorical_features, drop_features,
        LogisticRegression, param_dist_logreg
    )
    logreg_search.fit(X_train, y_train)
    assert isinstance(logreg_search, RandomizedSearchCV)
    assert hasattr(logreg_search, "best_score_")
    assert hasattr(logreg_search, "best_params_")
    assert isinstance(logreg_search.best_score_, float)
    assert isinstance(logreg_search.best_params_, dict)

def test_run_model_with_random_search_rfclf():
    param_dist_rfclf = {"randomforestclassifier__n_estimators": 50 * np.array([1, 2, 4]),
                        "randomforestclassifier__max_depth": [5, 10, 20, None],
                        "randomforestclassifier__max_features": ['sqrt', 'log2']}
    rfclf_search = run_model_with_random_search(
        X_train, y_train, numerical_features, credit_feature, categorical_features, drop_features,
        RandomForestClassifier, param_dist_rfclf
    )
    rfclf_search.fit(X_train, y_train)
    assert isinstance(rfclf_search, RandomizedSearchCV)
    assert hasattr(rfclf_search, "best_score_")
    assert hasattr(rfclf_search, "best_params_")
    assert isinstance(rfclf_search.best_score_, float)
    assert isinstance(rfclf_search.best_params_, dict)

def test_run_model_with_random_search_gbclf():
    param_dist_gbclf = {"gradientboostingclassifier__n_estimators": 50 * np.array([1, 2, 4]),
                        "gradientboostingclassifier__max_depth": [3, 5, 10],
                        "gradientboostingclassifier__learning_rate": [0.05, 0.1, 0.2]}
    gbclf_search = run_model_with_random_search(
        X_train, y_train, numerical_features, credit_feature, categorical_features, drop_features,
        GradientBoostingClassifier, param_dist_gbclf, need_class_weights=False
    )
    gbclf_search.fit(X_train, y_train)
    assert isinstance(gbclf_search, RandomizedSearchCV)
    assert hasattr(gbclf_search, "best_score_")
    assert hasattr(gbclf_search, "best_params_")
    assert isinstance(gbclf_search.best_score_, float)
    assert isinstance(gbclf_search.best_params_, dict)
