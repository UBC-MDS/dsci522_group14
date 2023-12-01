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
    drop_features, model_class, param_dist, n_iter=5, cv=2, scoring="f1", need_class_weights=True):
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
    if need_class_weights:
        pipe = make_pipeline(ct, model_class(class_weight="balanced"))
    else:
        pipe = make_pipeline(ct, model_class())
    random_search = RandomizedSearchCV(
        pipe, param_dist, n_jobs=-1, n_iter=n_iter, cv=cv, scoring=scoring, return_train_score=True
    )
    
    return random_search





