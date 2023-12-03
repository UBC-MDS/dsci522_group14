import os
import sys
import click
import pandas as pd
from io import BytesIO
import pickle
from io import StringIO
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.model import run_model_with_random_search

# Libraries needed for the modeling step
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
@click.command()
@click.option('--x_train_path', type=str, help="Path to X_train data")
@click.option('--y_train_path', type=str, help="Path to y_train data")
@click.option('--ct_path', type=str, help="Path to column transformer pickle data")
@click.option('--seed', type=int, default=123, help="Random seed")
@click.option('--model_save_path',type=str,help='Path of the model file to save')
# @click.option('--search_save_path',type=str,help='Path of the RandomizedSearchCV file to save')
@click.option('--table_to', type=str, help="Path to directory where the table will be written to")
@click.option('--plot_to', type=str, help="Path to directory where the plot will be written to")
def main(
    x_train_path, y_train_path, ct_path, seed, model_save_path, table_to, plot_to
):
    with open(x_train_path, 'rb') as file:
        X_train = pd.read_pickle(x_train_path)
    
    with open(y_train_path, 'rb') as file:
        y_train = pd.read_pickle(y_train_path)


    credit_feature=['creditLimit']
    drop_features=['transactionDateTime', 'currentExpDate', 'dateOfLastAddressChange']
    numerical_features=['availableMoney', 'transactionAmount', 'currentBalance']
    categorical_features=[
        'accountNumber','acqCountry','merchantCountryCode','posEntryMode','posConditionCode',
        'merchantCategoryCode','transactionType','cardPresent','expirationDateKeyInMatch','CVVmatched']
    
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
    with open(ct_path, 'rb') as file:
        ct = pickle.load(file)
    param_dist_logreg = {"logisticregression__C": 10.0 ** np.arange(-2, 3),
                        "logisticregression__solver": ["newton-cholesky", "lbfgs"]}
    if True:
    # if need_class_weights:
        pipe = make_pipeline(ct, LogisticRegression(class_weight="balanced"))
    else:
        pipe = make_pipeline(ct, LogisticRegression())
    random_search = RandomizedSearchCV(
        pipe, param_dist_logreg, n_jobs=-1, n_iter=100, cv=5, scoring='f1', random_state=seed, return_train_score=True
    )
    random_search.fit(X_train, y_train)
    model = random_search.best_estimator_
    

    with open(model_save_path, 'wb') as file:
        pickle.dump(model, file)

    search_result=pd.DataFrame(random_search.cv_results_)[["rank_test_score", 'mean_test_score', 'mean_train_score', 
                                         "param_logisticregression__C", "param_logisticregression__solver", 
                                         "mean_fit_time"]].sort_values("rank_test_score").set_index("rank_test_score")

    search_result.to_csv(table_to)

    plot_result=ConfusionMatrixDisplay.from_estimator(random_search, X_train, y_train, values_format='d')
   
    # Plot the confusion matrix
    plot_result.plot(cmap='Blues', values_format='d')

    # Save the plot to a PNG file
    plot_result.savefig(plot_to)

if __name__ == '__main__':
    main()