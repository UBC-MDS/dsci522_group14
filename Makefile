# adapted from https://github.com/ttimbers/breast_cancer_predictor_py/blob/v3.0.0/Makefile by Tiffany Timbers
all: report/_build/html/index.html

# download and extract data
data/transactions.pkl.zip : scripts/download_data.py
	python scripts/download_data.py \
        --url="https://github.com/CapitalOneRecruiting/DS/blob/173ca4399629f1e4e74146107eb9bef1e7009741/transactions.zip?raw=true" \
        --write-to=data/transactions.pkl.zip

# perform eda and save plots
results/plots/cat_plots.png results/plots/num_plots.png data/preprocessed/eda_processed.pkl : scripts/eda.py data/transactions.pkl.zip
	python scripts/eda.py \
        --df-path=data/transactions.pkl.zip \
        --save-to=results/plots \
        --write-to=data/preprocessed/eda_processed.pkl


# preprocess the data and save preprocessor
data/preprocessed/X_train.pkl data/preprocessed/y_train.pkl data/preprocessed/X_test.pkl data/preprocessed/y_test.pkl data/preprocessed/transformed_X_train.pkl data/transformers/ct.pkl results/tables/count_df.csv : scripts/preprocessing_data.py data/preprocessed/eda_processed.pkl
	python scripts/preprocessing_data.py \
        --df-path=data/preprocessed/eda_processed.pkl \
        --write-to=data \
        --table-to=results/tables/count_df.csv

# train model, create visualize tuning, and save plot and model
results/plots/confusion_matrix.png results/plots/precision_recall.png results/plots/score_comparison.png results/tables/model_table.csv : data/preprocessed/X_train.pkl data/preprocessed/y_train.pkl data/preprocessed/X_test.pkl data/preprocessed/y_test.pkl data/preprocessed/transformed_X_train.pkl data/transformers/ct.pkl
	python scripts/model.py \
        --df-path=data/preprocessed \
        --ct-path=data/transformers/ct.pkl \
        --table-to=results/tables/model_table.csv\
        --plot-to=results/plots

# build HTML report and copy build to docs folder
report/_build/html/index.html : report/fraud_detection_full.ipynb \
report/references.bib \
report/_toc.yml \
report/_config.yml \
results/plots/confusion_matrix.png \
results/plots/precision_recall.png \
results/plots/score_comparison.png \
results/tables/model_table.csv
	jupyter-book build report
	cp -r report/_build/html/* docs

# clean up analysis
clean :
	rm -f data/transactions.pkl.zip
	rm -rf data/preprocessed/*
	rm -f data/transformers/ct.pkl
	rm -rf results/plots/*
	rm -rf results/tables/*
	rm -rf report/_build
	rm -rf docs/*
