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
from sklearn.metrics import ConfusionMatrixDisplay
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

    print(X_train.columns.tolist())

    with open(ct_path, 'rb') as ct_file:
        ct = pickle.load(ct_file)

    param_dist_logreg = {"logisticregression__C": 10.0 ** np.arange(-2, 3),
                        "logisticregression__solver": ["newton-cholesky", "lbfgs"]}
    if True:
    # if need_class_weights:
        pipe = make_pipeline(ct, LogisticRegression(class_weight="balanced"))
    else:
        pipe = make_pipeline(ct, LogisticRegression())

    random_search = RandomizedSearchCV(
        pipe, param_dist_logreg, n_jobs=-1, n_iter=100, cv=5, scoring='f1', 
        random_state=522, return_train_score=True
    )

    random_search.fit(X_train, y_train)
    model = random_search.best_estimator_
    
    # with open(model_save_path, 'wb') as file:
    #     pickle.dump(model, file)

    search_result=pd.DataFrame(random_search.cv_results_)[["rank_test_score", 'mean_test_score', 'mean_train_score', 
                                                           "param_logisticregression__C", "param_logisticregression__solver", 
                                                           "mean_fit_time"]].sort_values("rank_test_score").set_index("rank_test_score")

    search_result.to_csv(table_to)

    plot_result=ConfusionMatrixDisplay.from_estimator(random_search, X_train, y_train, values_format='d')
   
    # Plot the confusion matrix
    cm_fig = plot_result.plot(cmap='Blues', values_format='d')
    cm_fig.plot()

    plt.savefig(plot_to)

    # Save the plot to a PNG file
    # plot_result.savefig(plot_to)

if __name__ == '__main__':
    main()