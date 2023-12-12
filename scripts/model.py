import os
import sys
import click
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import pickle
from io import StringIO
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Libraries needed for the modeling step

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, f1_score
from sklearn.compose import ColumnTransformer

@click.command()
@click.option('--df-path', type=str, help="Path to X and y trained data")
@click.option('--ct-path', type=str, help="Path to column transformer pickle data")
@click.option('--table-to', type=str, help="Path to directory where the table will be written to")
@click.option('--plot-to', type=str, help="Path to directory where the plot will be written to")

# python scripts/model.py --df-path data/preprocessed --ct-path data/transformers/ct.pkl --table-to data/preprocessed/model_table.csv --plot_to visualization/final_plot.png

def main(
    df_path, ct_path, table_to, plot_to
):

    X_train = pd.read_pickle(f"{df_path}/X_train.pkl", compression="zip")
    y_train = pd.read_pickle(f"{df_path}/y_train.pkl", compression="zip")
    X_test = pd.read_pickle(f"{df_path}/X_test.pkl", compression="zip")
    y_test = pd.read_pickle(f"{df_path}/y_test.pkl", compression="zip")

    print(X_train.columns.tolist())

    with open(ct_path, 'rb') as ct_file:
        ct = pickle.load(ct_file)

    param_dist_logreg = {"logisticregression__C": 10.0 ** np.arange(-2, 3),
                        "logisticregression__solver": ["newton-cholesky", "lbfgs"]}

    pipe = make_pipeline(ct, LogisticRegression(class_weight="balanced"))

    random_search = RandomizedSearchCV(
        pipe, param_dist_logreg, n_jobs=-1, n_iter=100, cv=5, scoring='f1', 
        random_state=522, return_train_score=True
    )

    random_search.fit(X_train, y_train)

    search_result=pd.DataFrame(random_search.cv_results_)[["rank_test_score", 'mean_test_score', 'mean_train_score', 
                                                           "param_logisticregression__C", "param_logisticregression__solver", 
                                                           "mean_fit_time"]].sort_values("rank_test_score").set_index("rank_test_score")

    search_result.to_csv(table_to)

    train_scores = []
    test_scores = []
    C_list = 10.0 ** np.arange(-4, 3)
    for c in C_list:
        pipe = make_pipeline(ct, LogisticRegression(class_weight="balanced", C=c))
        pipe.fit(X_train, y_train)
        X_train_pred = pipe.predict(X_train)
        train_scores.append(f1_score(X_train_pred, y_train))
        X_test_pred = pipe.predict(X_test)
        test_scores.append(f1_score(X_test_pred, y_test))
        print(f"{c} done")

    plt.plot(C_list, train_scores, label='train score')
    plt.plot(C_list, test_scores, label='test score')
    plt.xscale("log")
    # Adding a legend
    plt.legend()
    plt.title("Train and Test Score Comparison on Logistic Regression")
    plt.savefig(f"{plot_to}/score_comparison.png")

    plot_result=ConfusionMatrixDisplay.from_estimator(random_search, X_train, y_train, values_format='d')
   
    # Plot the confusion matrix
    cm_fig = plot_result.plot(cmap='Blues', values_format='d')
    cm_fig.plot()

    plt.title("Consufion Matrix on Logistic Regression")
    plt.savefig(f"{plot_to}/confusion_matrix.png")

    pr_curve = PrecisionRecallDisplay.from_estimator(
        random_search, X_train, y_train, name="Logistic Regression")
    pr_curve.ax_.set_title("Precision-Recall Curve for Test Data")
    plt.savefig(f"{plot_to}/precision_recall.png")


if __name__ == '__main__':
    main()