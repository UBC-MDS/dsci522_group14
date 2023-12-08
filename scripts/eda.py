import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
import click
import sys

warnings.filterwarnings('ignore')

@click.command()
@click.option('--df-path', type=str, help="Path to get the downloaded dataset")
@click.option('--save-to', type=str, help="Path to save the plots")
@click.option('--write-to', type=str, help="Filename to write the processed data")
def main(df_path, save_to, write_to):
    df = pd.read_pickle(df_path, compression="infer")
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = df.select_dtypes(include=['object', 'bool']).columns

    # Correct calculation for the number of rows needed for numerical features
    num_rows_num = 2 * len(numerical_features)//4
    fig_num, axes_num = plt.subplots(num_rows_num, 4, figsize=(20, 5 * num_rows_num))
    axes_num = axes_num.ravel()

    for i, col in enumerate(numerical_features):
        sns.histplot(df[col], ax=axes_num[2*i], kde=False, bins=30)
        axes_num[2*i].set_title(f'Histogram of {col}')
        axes_num[2*i].set_ylabel('Count')
        sns.boxplot(df[col], ax=axes_num[2*i + 1])
        axes_num[2*i + 1].set_title(f'Boxplot of {col}')
        axes_num[2*i + 1].set_ylabel('')
    fig_num.suptitle('Numerical Features: Histograms and Box Plots', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_to}/num_plots.png')

    num_rows_cat = (len(categorical_features) + 3) // 4
    fig_cat, axes_cat = plt.subplots(num_rows_cat, 4, figsize=(20, 5 * num_rows_cat))
    axes_cat = axes_cat.ravel()

    for j, col in enumerate(categorical_features):
        counts = df[col].value_counts().nlargest(10)
        sns.barplot(x=counts.index, y=counts.values, ax=axes_cat[j])
        axes_cat[j].set_title(f'Frequency of Top 10 {col}')
        axes_cat[j].set_xticklabels(axes_cat[j].get_xticklabels(), rotation=45)
        axes_cat[j].set_ylabel('Count')
    fig_cat.suptitle('Categorical Features: Frequency Plots', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_to}/cat_plots.png')

    # Ensure the write_to path includes a filename, e.g., 'processed_data.pkl'
    df.to_pickle(write_to)
if __name__ == "__main__":
	main()
