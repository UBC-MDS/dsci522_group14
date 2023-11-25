import pandas as pd
import pytest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.load_data import load_data


def test_return_type():
    df = load_data()
    assert isinstance(df, pd.DataFrame), "Returning incorrect data type a pandas dataframe should be returned."


def test_return_type_none():
    old_name = '../data/transactions.pkl.zip'
    temp_name = '../data/temp.pkl.zip'
    os.rename(old_name, temp_name)
    df = load_data()
    assert df is None, "Returning incorrect data type None should be returned."
    os.rename(temp_name, old_name)


def test_number_of_rows():
    df = load_data()
    assert len(df) == 786363, "Incorrect number of rows in the dataframe."


def test_number_of_columns():
    df = load_data()
    assert len(df.columns) == 29, "Incorrect number of columns in the dataframe."


def test_column_names():
    df = load_data()
    assert df.columns.tolist() == ['accountNumber', 'customerId', 'creditLimit', 'availableMoney',
                                   'transactionDateTime', 'transactionAmount', 'merchantName',
                                   'acqCountry', 'merchantCountryCode', 'posEntryMode',
                                   'posConditionCode', 'merchantCategoryCode', 'currentExpDate',
                                   'accountOpenDate', 'dateOfLastAddressChange', 'cardCVV',
                                   'enteredCVV', 'cardLast4Digits', 'transactionType',
                                   'echoBuffer', 'currentBalance', 'merchantCity', 'merchantState',
                                   'merchantZip', 'cardPresent', 'posOnPremises', 'recurringAuthInd',
                                   'expirationDateKeyInMatch', 'isFraud'], "Incorrect column names in the dataframe."
