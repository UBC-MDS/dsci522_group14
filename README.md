# Credit Card Fraud Detection

-   Author: Jenny Lee, Shawn X. Hu, Koray Tecimer, Iris Luo

Contains an analytical report generated for Milestone 1 of DSCI 522, a course offered through the Master of Data Science program at the University of British Columbia.

## About

Through this project, we attempted to build three classification models capable of distinguishing between fraud and non-fraud transactions, as indicated on customer accounts. Data preprocessing steps are also described in `fraud_detection.ipynb`. The models we experimented with include logistic regression, random forest classifier, and gradient boost classifier. Due to an extreme imbalance in our data, we encountered challenges in developing an effective model in milestone 1. In the discussion, we have provided some suggestions for future steps.

## Data

Our data is sourced from the [Capital One GitHub for Data Scientist Recruitment](https://github.com/CapitalOneRecruiting/DS). The data consists of 786,363 entries of synthetically generated data.

Our dataset exhibits a significant imbalance. The imbalanced nature of the data significantly affected the performance of our models, preventing them from achieving a high scoring metric value (f1).

## Report

Our final report can be found here. 
- Our final report can be found [here](https://ubc-mds.github.io/fraud_detection/fraud_detection_full.html). 

## Quick Start (Docker)

Navigate to project folder

```         
cd/to/fraud_detection
```

Run:

```         
docker-compose up
```

Locate your url with token from the log and paste it in your browser to access container and project. Should be something like `http://127.0.0.1:8888/lab?token=token_hash`. 

To run the analysis you can run the commands below:
```
# download and extract data
python scripts/download_data.py \
   --url="https://github.com/CapitalOneRecruiting/DS/blob/173ca4399629f1e4e74146107eb9bef1e7009741/transactions.zip?raw=true" \
   --write-to=data/transactions.pkl.zip

# Exploratory data analysis and data wrangling
python scripts/eda.py \
   --df-path=data/transactions.pkl.zip \
   --save-to=results/plots \
   --write-to=data/preprocessed/eda_processed.pkl

# Perform additional data preprocessing to avoid curse of dimensionality
python scripts/preprocessing_data.py \
   --df-path=data/preprocessed/eda_processed.pkl \
   --write-to=data \
   --table-to=results/tables/count_df.csv

# train model, create visualize tuning, and save plot and model
python scripts/model.py \
   --df-path=data/preprocessed \
   --ct-path=data/transformers/ct.pkl \
   --table-to=results/tables/model_table.csv \
   --plot-to=results/plots

# build HTML report and copy build to docs folder
jupyter-book build report
cp -r report/_build/html/* docs
```

## Clean-Up
After finishing with a working session, copy and paste the following code below on your terminal to clean up the working space.
```
docker compose down
```

## Tests
To run the tests, navigate to project folder:

```         
cd/to/fraud_detection
```

Run:

```         
pytest
```

Or run from **root directory**:
```
tests/pytest
```

## Dependencies

-   `conda` (version 23.7.4 or higher)
-   `nb_conda_kernels` (version 2.3.1 or higher)
-   Python packages listed in `environment.yml`, including:
```
- pandas=1.3.2
- scikit-learn=1.3.2
- numpy=1.21.1
- matplotlib=3.4.3
- seaborn=0.11.2
- pytest=7.4.3
- click=8.1.7
```

## License

Licenses used in this project are listed below. More detailed information can be found at `LICENSE.md`. - MIT License - Copyright (c) 2023 Master of Data Science at the University of British Columbia


## Disclaimer

The overall format of `README.md` is retrieved from the [sample project repository](https://github.com/ttimbers/breast_cancer_predictor_py/tree/0.0.1).

## References

Capital One. (2018). Capital One Data Science Challenge. In *CapitalOneRecruiting GitHub Repository*. <https://github.com/CapitalOneRecruiting/DS>

Python Software Foundation. Python Language Reference, version 3.11.6. Available at <http://www.python.org>

Timbers, T., Lee, M. & Ostblom, J. (2023). Breast Cancer Predictor. <https://github.com/ttimbers/breast_cancer_predictor_py/tree/0.0.1>

Pedregosa, F., Varoquaux, Ga"el, Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... others. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12(Oct), 2825--2830

McKinney, W., & others. (2010). Data structures for statistical computing in python. In Proceedings of the 9th Python in Science Conference (Vol. 445, pp. 51--56).

Caporal, J. (2023). Identity Theft and Credit Card Fraud Statistics for 2023. In *the Ascent*. <https://www.fool.com/the-ascent/research/identity-theft-credit-card-fraud-statistics/>
