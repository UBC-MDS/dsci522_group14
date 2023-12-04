import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import click 
##

@click.command()
@click.option('--df-path', type=str, help="Path to get the downloaded dataset")
@click.option('--path', type=str, help="Path to directory where raw data will be written to")
def main(df_path, path):
    """
    Perform a full exploratory data analysis (EDA) on the given DataFrame.

    Parameters:
    df_path (str): The path to the DataFrame to be processed.
    path (str): The path to directory where plots will be saved.

    Returns:
    None: This function only displays plots and does not return any value.
    """
    df = pd.read_pickle(df_path, compression="infer")
    # Process the DataFrame by counting and removing empty strings
    empty_string_counts = df.apply(lambda column: (column == '').sum())
    print(empty_string_counts)

    columns_to_drop = [col for col, count in empty_string_counts.items() if count > 50000]
    columns_to_drop.extend(['echoBuffer', 'merchantCity', 'merchantZip', 'posOnPremises', 'recurringAuthInd', 'merchantState'])
    df.drop(columns=columns_to_drop, axis=1, inplace=True)
    directory = 'data/preprocessed/raw_df.pkl'
    df.to_pickle(directory)
    empty_string_counts = df.apply(lambda column: (column == '').sum())
    print(empty_string_counts)
    # Selecting numerical and categorical features
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = df.select_dtypes(include=['object', 'bool']).columns

    # Plotting for numerical features
    num_plots_num = len(numerical_features) * 2  # Two plots (histogram and boxplot) for each numerical feature
    num_rows_num = num_plots_num // 2
    fig_num, axes_num = plt.subplots(num_rows_num, 2, figsize=(15, 5 * num_rows_num))
    axes_num = axes_num.ravel()

    for i, col in enumerate(numerical_features):
        sns.histplot(df[col], ax=axes_num[2*i], kde=False, bins=30)
        axes_num[2*i].set_title(f'Histogram of {col}')
        axes_num[2*i].set_ylabel('Count')
        sns.boxplot(data=df, x=col, ax=axes_num[2*i + 1])
        axes_num[2*i + 1].set_title(f'Box plot of {col}')

    plt.tight_layout()  # Adjusts the plots to fit into the figure neatly
    plt.savefig('data/num_plots.png')

    # Plotting for categorical features
    num_plots_cat = len(categorical_features)  # One plot for each categorical feature
    fig_cat, axes_cat = plt.subplots(num_plots_cat, 1, figsize=(15, 5 * num_plots_cat))
    axes_cat = axes_cat.ravel() if num_plots_cat > 1 else [axes_cat]

    for j, col in enumerate(categorical_features):
        counts = df[col].value_counts().nlargest(10)
        sns.barplot(x=counts.index, y=counts.values, ax=axes_cat[j])
        axes_cat[j].set_title(f'Frequency of Top 10 {col}')
        axes_cat[j].set_xticklabels(axes_cat[j].get_xticklabels(), rotation=45)
        axes_cat[j].set_ylabel('Count')

    plt.tight_layout()  # Adjusts the plots to fit into the figure neatly
    plt.savefig('data/cat_plots.png')
    
# Example usage
if __name__ == "__main__":
  
    main()
