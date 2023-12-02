import os
import sys
import click
import pandas as pd
from io import BytesIO
import pickle
from io import StringIO
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.preprocessing import generalize_categories, count_unique_numbers

# Libraries needed for the preprocessing step
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler 
from sklearn.pipeline import make_pipeline

@click.command()
@click.option('--df-path', type=str, help="Path to retreive processed data from EDA")
@click.option('--path', type=str, help="Path to directory where preprocessed data will be written to")

def main(df_path, path):
      df = load_data(df_path)
    
      categorical_features = ['accountNumber',
                  'customerId',
                  'merchantName',
                  'acqCountry',
                  'merchantCountryCode',
                  'posEntryMode',
                  'posConditionCode',
                  'merchantCategoryCode',
                  'currentExpDate',
                  'accountOpenDate',
                  'dateOfLastAddressChange',
                  'cardLast4Digits',
                  'transactionType',
                  'cardPresent',
                  'expirationDateKeyInMatch',
                  'isFraud',
                  'CVVmatched'
                  ]
      
      count_df = count_unique_numbers(df, categorical_features)
      count_df.to_pickle(path+"/preprocessed/count_df.pkl", compression='zip')
    
      numerical_features = ['creditLimit', 'availableMoney', 'transactionAmount', 'currentBalance']

      drop_features = ["transactionDateTime"]
      df = df.drop(["accountOpenDate", 'cardLast4Digits', 'customerId', 'merchantName'], axis=1)

      categorical_features.remove("isFraud")
      categorical_features.remove("customerId")
      categorical_features.remove("accountOpenDate")
      categorical_features.remove("cardLast4Digits")
      categorical_features.remove("merchantName")
      categorical_features.remove("currentExpDate")
      categorical_features.remove("dateOfLastAddressChange")

      drop_features.append("currentExpDate")
      drop_features.append("dateOfLastAddressChange")

      preprocess_df = perform_category_generalization(df, categorical_features)
      numerical_features.remove("creditLimit")
      credit_feature = ["creditLimit"]

      train_df, test_df = train_test_split(preprocess_df, test_size=0.2, random_state=522)
      X_train, y_train = train_df.drop(["isFraud"], axis=1), train_df["isFraud"]
      X_test, y_test = test_df.drop(["isFraud"], axis=1), test_df["isFraud"]

      ct = column_transformer(categorical_features, numerical_features, credit_feature, drop_features)

      with open(path+"/transformers/ct.pkl", 'wb') as pickle_file:
            pickle.dump(ct, pickle_file)

      X_train.to_pickle(path+"/preprocessed/X_train.pkl", compression='zip')
      X_test.to_pickle(path+"/preprocessed/X_test.pkl", compression='zip')
      y_train.to_pickle(path+"/preprocessed/y_train.pkl", compression='zip')
      y_test.to_pickle(path+"/preprocessed/y_test.pkl", compression='zip')

      transformed_df = ct.fit_transform(X_train)

      column_names = (
            numerical_features
            + credit_feature
            + ct.named_transformers_["onehotencoder"].get_feature_names_out().tolist()
            )
      
      transformed_X_train = pd.DataFrame(transformed_df.toarray(), columns=column_names)
      
      transformed_X_train.to_pickle(path+"/preprocessed/transformed_X_train.pkl", compression='zip')


def column_transformer(
            categorical_features,
            numerical_features, 
            credit_feature, 
            drop_features):
     
     ct = make_column_transformer(
          (
               make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
               numerical_features
        ),
        (
             make_pipeline(SimpleImputer(strategy="mean"), StandardScaler()),
             credit_feature,
        ),
        (
             OneHotEncoder(drop="if_binary", handle_unknown="ignore"), 
             categorical_features),
        ("drop", drop_features)
     )
     return ct

def perform_category_generalization(df, categorical_features):
      preprocess_df = df.copy()
      for category_column in categorical_features:
            preprocess_df[category_column] = preprocess_df[category_column].replace('', 'Other')
            if len(preprocess_df[category_column].unique().tolist()) > 10:
                  preprocess_df = generalize_categories(category_column, preprocess_df)
      return preprocess_df

def load_data(df_path):
    return pd.read_pickle(df_path, compression="infer")

if __name__ == '__main__':
    main()