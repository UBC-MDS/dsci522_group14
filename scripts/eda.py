import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import click 
@click.command()
@click.option('--df-path', type=str, help="Path to get the downloaded dataset")
@click.option('--path', type=str, help="Path to directory where raw data will be written to")
#shawn
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
    df_drop=df.drop(columns=columns_to_drop, axis=1, inplace=False)
    directory = 'data/preprocessed/raw_df.pkl'
    df_drop.to_pickle(directory)


    # Selecting numerical and categorical features
    numerical_features = df_drop.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = df_drop.select_dtypes(include=['object', 'bool']).columns	
    total_plots = len(numerical_features) * 2 + len(categorical_features)  # Two plots for each numerical feature (histogram and boxplot) and one for each categorical feature
    num_rows = (total_plots + 1) // 2  # Calculate the number of rows needed
    fig, axes = plt.subplots(num_rows, 2, figsize=(15, 5 * num_rows))	
    axes = axes.ravel()
    for i, col in enumerate(numerical_features):
    	sns.histplot(df_drop[col], ax=axes[2*i], kde=False, bins=30)
    	axes[2*i].set_title(f'Histogram of {col}')
    	axes[2*i].set_ylabel('Count')
    	sns.boxplot(data=df_drop, x=col, ax=axes[2*i + 1])
    	axes[2*i + 1].set_title(f'Box plot of {col}')
    start_index = 2 * len(numerical_features)
    for j, col in enumerate(categorical_features, start=start_index):
    	counts = df_drop[col].value_counts().nlargest(10)
    	sns.barplot(x=counts.index, y=counts.values, ax=axes[j])
    	axes[j].set_title(f'Frequency of Top 10 {col}')
    	axes[j].set_xticklabels(axes[j].get_xticklabels(), rotation=45)
    	axes[j].set_ylabel('Count')
    plt.tight_layout()
    path = './data/'  # Adjust the path as necessary
    plt.savefig('data/combined_plots.png')
    

# Example usage
if __name__ == "__main__":
    main()
