# Notebook Organization

There are three notebooks included in this repository. They are differentiated for the following reasons:

#### `fraud_detection_full.ipynb`
The final notebook that will be used to build the Jupyter book. Whenever a change is made in this notebook, you must also make the same changes in `report/fraud_detection_full.ipynb` if you wish those changes to be reflected on the Jupyter book.

#### `fraud_detection.ipynb`
This notebook is a code-heavy notebook which was used for the Milestone 1 and Milestone 2 of DSCI 522. All codes used in this notebook have been transferred over to `scripts`, but if you would like to see all lines of code in a single notebook, please refer to this.

#### `generate_sample_data.ipynb`
This notebook is used to generate a sample data that is used in our `pytest` tests. Its output is stored in `data/preprocessed/sample_df.csv`. 