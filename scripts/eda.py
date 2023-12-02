import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import click 
@click.command()
@click.option('--df-path', type=str, help="Path to get the downloaded dataset")
@click.option('--path', type=str, help="Path to directory where raw data will be written to")

def main(df_path, path):
    '''
    Perform a full exploratory data analysis (EDA) on the given DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame to be processed.

    Returns:
    None: This function only displays plots and does not return any value.
    '''
    df = pd.read_pickle(df_path, compression="infer")
    # Process the DataFrame by counting and removing empty strings
    empty_string_counts = df.apply(lambda column: (column == '').sum())
    print(empty_string_counts)

    columns_to_drop = [col for col, count in empty_string_counts.items() if count > 50000]
    columns_to_drop.extend(['echoBuffer', 'merchantCity', 'merchantZip', 'posOnPremises', 'recurringAuthInd', 'merchantState'])
    df.drop(columns=columns_to_drop, axis=1, inplace=True)

    # Selecting numerical and categorical features
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = df.select_dtypes(include=['object', 'bool']).columns

    # Creating feature plots
    total_plots = len(numerical_features) + len(categorical_features)
    num_rows = (total_plots + 1) // 2

    fig, axes = plt.subplots(num_rows, 2, figsize=(15, 5 * num_rows))
    axes = axes.ravel()

    for i, col in enumerate(numerical_features):
        sns.histplot(df[col], ax=axes[i], kde=False, bins=30)
        axes[i].set_title(f'Histogram of {col}')
        axes[i].set_ylabel('Count')

    for j, col in enumerate(categorical_features, start=len(numerical_features)):
        counts = df[col].value_counts().nlargest(10)
        sns.barplot(x=counts.index, y=counts.values, ax=axes[j])
        axes[j].set_title(f'Frequency of Top 10 {col}')
        axes[j].set_xticklabels(axes[j].get_xticklabels(), rotation=45)
        axes[j].set_ylabel('Count')

    plt.tight_layout()
    plt.savefig(path + 'plot1.png')

    # Creating box plots for numerical features
    for col in numerical_features:
        plt.figure(figsize=(8, 4))
        sns.boxplot(data=df, x=col)
        plt.title(f'Box plot of {col}')
        plt.savefig(path + 'plot2.png')

    

# Example usage
if __name__ == "__main__":
    main()
