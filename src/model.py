import pandas as pd
import numpy as np
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV

# Acknowledgement:
# The format of the function documentations are derived from the sample project provided. 
# Proper citation to the sample projct is included in `notebooks/fraud_detection.ipynb`. 
# https://github.com/ttimbers/demo-tests-ds-analysis-python/blob/main/src/count_classes.py
# Proper citation to the sample projct is included in `UGFraud`
# https://github.com/safe-graph/UGFraud/blob/master/UGFraud/Detector/SVD.py

def run_model_with_random_search(
    X_train, y_train, numerical_features, credit_feature, categorical_features, 
    drop_features, model_class, param_dist, n_iter=5, cv=2, scoring="f1"):
    """
    Run a machine learning model with hyperparameter tuning using RandomizedSearchCV.

    Parameters:
    - X_train: Training input data.
    - y_train: True labels for training data.
    - numerical_features: List of numerical feature names.
    - credit_feature: Credit feature name.
    - categorical_features: List of categorical feature names.
    - drop_features: List of features to drop.
    - model_class: Machine learning model class (e.g., LogisticRegression, RandomForestClassifier).
    - param_dist: Parameter distribution for RandomizedSearchCV.
    - n_iter: Number of iterations for RandomizedSearchCV.
    - cv: Number of cross-validation folds.
    - scoring: Scoring metric for RandomizedSearchCV.

    Returns:
    - RandomizedSearchCV object

    Example:
    ```
    param_dist_logreg = {"logisticregression__C": 10.0 ** np.arange(-2, 3),
                        "logisticregression__solver": ["newton-cholesky", "lbfgs"]}
    logreg_search = run_model_with_random_search(
    X_train, y_train, numerical_features, credit_feature, categorical_features, drop_features, 
    LogisticRegression, param_dist_logreg)
    logreg_search.fit(X_train, y_train)
    ```
    """
    # Input validation
    if not isinstance(X_train, pd.DataFrame):
        raise ValueError("X_train must be a DataFrame.")
    if not isinstance(y_train, pd.Series):
        raise ValueError("y_train must be a Series.")
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError("Number of rows in X_train and y_train must match.")
    if X_train.shape[1] != len(numerical_features) + len(credit_feature) + len(categorical_features) + len(drop_features):
        raise ValueError("Mismatch in the number of features.")
    
    # Check for duplicate feature names
    all_features = set(numerical_features + credit_feature + categorical_features + drop_features)
    if len(all_features) != len(numerical_features) + len(credit_feature) + len(categorical_features) + len(drop_features):
        raise ValueError("Duplicate feature names found.")
    
    # Ensure all features are non-empty strings
    if not all(isinstance(feature, str) and feature for feature in all_features):
        raise ValueError("All feature names must be non-empty strings.")
    
    # Check for overlapping features between numerical and categorical features
    if any(feature in numerical_features for feature in categorical_features):
        raise ValueError("Numerical and categorical features must not overlap.")
    
    # Check if numerical and credit features have any common elements
    if any(feature in numerical_features for feature in credit_feature):
        raise ValueError("Numerical and credit features must not have common elements.")

    
    ct = make_column_transformer(
        (
        make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
        numerical_features,
        ),
        (
        make_pipeline(SimpleImputer(strategy="mean"), StandardScaler()),
        credit_feature,
        ),
        (OneHotEncoder(drop="if_binary", handle_unknown="ignore"), categorical_features),
        ("drop", drop_features)
    )
    
    pipe = make_pipeline(ct, model_class(class_weight="balanced"))
    
    random_search = RandomizedSearchCV(
        pipe, param_dist, n_jobs=-1, n_iter=n_iter, cv=cv, scoring=scoring, return_train_score=True
    )
    
    return random_search

param_dist_logreg = {"logisticregression__C": 10.0 ** np.arange(-2, 3),
                     "logisticregression__solver": ["newton-cholesky", "lbfgs"]}
logreg_search = run_model_with_random_search(X_train, y_train, numerical_features, credit_feature, categorical_features, drop_features, LogisticRegression, param_dist_logreg)
logreg_search.fit(X_train, y_train)

param_dist_rfclf = {"randomforestclassifier__n_estimators": 50 * np.array([1, 2, 4]),
                   "randomforestclassifier__max_depth": [5, 10, 20, None],
                   "randomforestclassifier__max_features": ['sqrt', 'log2']}
rfclf_search = run_model_with_random_search(X_train, y_train, numerical_features, credit_feature, categorical_features, drop_features, RandomForestClassifier, param_dist_rfclf)
rfclf_search.fit(X_train, y_train)

param_dist_gbclf = {"gradientboostingclassifier__n_estimators": 50 * np.array([1, 2, 4]),
                   "gradientboostingclassifier__max_depth": [3, 5, 10],
                   "gradientboostingclassifier__learning_rate": [0.05, 0.1, 0.2]}
gbclf_search = run_model_with_random_search(X_train, y_train, numerical_features, credit_feature, categorical_features, drop_features, GradientBoostingClassifier, param_dist_gbclf)
gbclf_search.fit(X_train, y_train)

