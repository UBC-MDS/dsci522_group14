# dsci522_group14

# Credit Card Fraud Detection

- Author: Jenny Lee, Shawn X. Hu, Koray Tecimer, Iris Luo

Contains analytical report generated for the milestone 1 of DSCI 522; a course offeredd through the Master of Data Science program at University of British Columbia. 

## About
Through this project, we built three classification models that can distinguish between fraud and non-fraud transactions marked on customer accounts. Data preprocessing steps are also included in `fraud_detection.ipynb`. The list of models that we tried are logistic regression, random forest classifier, and gradient boost classifier. Due to an extrememe imbalance in our data, we were not successful in building an effective model in milestone 1. We included some suggestions for future steps in discussion. 

### Data
Our data is retrieved from [Capital One GitHub for Data Scientist Recruitment](https://github.com/CapitalOneRecruiting/DS). The data comprises of 786363 entries of synthetically generated data. 

![isfraud](visualization/isfraud.png)

As seen from the graph above, our dataset is extremely unbalanced. Due to the imbalance in data our models suffered from achieving high scoring metric value (f1). 

### Usage
Packages with detailed versions used in this project can be found below. To run the project, copy and paste the code below on local terminal. 
```
conda env create —file environmentgroup14.yml
```
To run the project, copy and paste the commands below on local commands from your root directory.
```
conda activate 522group14
jupyter lab
```

### Dependencies
- `conda` (version 23.7.4 or higher)
- `nb_conda_kernels` (version 2.3.1 or higher)
- Python packages listed in `environmentgroup14.yml`.

#### Disclaimer
The overall format of `README.md` is retrieved from the [sample project repository](https://github.com/ttimbers/breast_cancer_predictor_py/tree/0.0.1). 

### References
- Capital One. (2018). Capital One Data Science Challenge. In *CapitalOneRecruiting GitHub Repository*. https://github.com/CapitalOneRecruiting/DS
- Python Software Foundation. Python Language Reference, version 3.11.6. Available at http://www.python.org
- Timbers, T., Lee, M. & Ostblom, J. (2023). Breast Cancer Predictor. https://github.com/ttimbers/breast_cancer_predictor_py/tree/0.0.1
- Pedregosa, F., Varoquaux, Ga"el, Gramfort, A., Michel, V., Thirion, B., Grisel, O., … others. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12(Oct), 2825–2830.
- McKinney, W., & others. (2010). Data structures for statistical computing in python. In Proceedings of the 9th Python in Science Conference (Vol. 445, pp. 51–56).