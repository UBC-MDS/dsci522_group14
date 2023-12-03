def eda_process(df):
    """
    Process the input DataFrame by counting the number of empty strings in each column, 
    printing these counts, and then dropping specified columns or any columns with more 
    than 50,000 empty strings.

    This function performs the following tasks:
    1. Counts and prints the number of empty strings ('') in each column of the DataFrame.
    2. Drops any columns with more than 50,000 empty strings.
    3. Additionally, drops a predefined set of columns from the DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame to be processed.

    Returns:
    pandas.DataFrame: The modified DataFrame with specified columns and any columns with 
                      excessive empty strings dropped.

    Note:
    This function modifies the input DataFrame in-place. Therefore, the original DataFrame 
    passed to this function will be altered.

    Example usage:
    >>> processed_df = process_dataframe(your_dataframe)
    """

    def count_empty_strings(column):
        return (column == '').sum()

    # Count empty strings in each column
    empty_string_counts = df.apply(count_empty_strings)

    # Print the counts of empty strings
    print(empty_string_counts)

    # Drop columns with more than 50,000 empty strings
    columns_to_drop = [col for col, count in empty_string_counts.items() if count > 50000]

    # Add additional specific columns to drop
    columns_to_drop.extend(['echoBuffer', 'merchantCity', 'merchantZip', 'posOnPremises', 'recurringAuthInd', 'merchantState'])

    # Drop the columns
    df.drop(columns=columns_to_drop, axis=1, inplace=True)

    return df