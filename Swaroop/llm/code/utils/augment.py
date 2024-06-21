import os
import pandas as pd
from typing import List, Union, Optional
from sklearn.utils import resample

def create_csv_from_subset(dataset_path: str, columns: List[str], output_path: str, output_name: str, verbose: int = 1) -> None:
    """
    Create a CSV file from a subset of columns in a dataset.

    Parameters:
        dataset_path (str): The path to the dataset file.
        columns (List[str]): List of columns to include in the subset.
        output_path (str): The path to save the output file.
        output_name (str): The name of the output file.
        verbose (int): Level of verbosity in output logging.

    Returns:
        None
    """

    # # for 1.csv

    # cols = ['name', 'dna_seq', 
    #         'deltaG_t', 'deltaG_t_95CI_high', 'deltaG_t_95CI_low', 'deltaG_t_95CI', 
    #         'deltaG_c', 'deltaG_c_95CI_high', 'deltaG_c_95CI_low', 'deltaG_c_95CI', 
    #         'deltaG', 'deltaG_95CI_high', 'deltaG_95CI_low', 'deltaG_95CI']
    
    # create_csv_from_subset(dataset_path='data/Processed_K50_dG_datasets/Tsuboyama2023_Dataset1_20230416.csv', columns=cols, output_path='data', output_name='ds1', verbose=1)

    try:
        df = pd.read_csv(dataset_path)

        if verbose > 0:
            print(f"Dataset loaded from {dataset_path}")
            print("Dataframe columns: ", df.columns)
            print(f"Initial shape: {df.shape}")
            print("Columns selected for output: ", columns)

        df_subset = df[columns]

        if verbose > 1:
            print("Preview of the dataset subset:")
            print(df_subset.head())

        output_full_path = os.path.join(output_path, f"{output_name}.csv")
        df_subset.to_csv(output_full_path, index=False)

        if verbose > 0:
            print(f"Subset CSV created at {output_full_path}")

    except Exception as e:
        if verbose > 0:
            print(f"An error occurred: {e}")

def resample_csv():
    df = pd.read_csv('../../data/ds23_sm.csv')

    # Check value counts and determine the maximum count to match
    max_size = df['deltaG'].value_counts().max()

    print(df['deltaG'].min())
    print(df['deltaG'].value_counts())

    # Define bins and labels for deltaG ranging from -16 to 17
    bins = list(range(-16, 18))  # Generates the bins from -16 to 17
    labels = [f'{i} to {i + 1}' for i in range(-16, 17)]

    # Categorize the deltaG column into these bins
    df['deltaG_bin'] = pd.cut(df['deltaG'], bins=bins, labels=labels, include_lowest=True)

    # Define the threshold
    threshold = 10

    # Initialize an empty list to collect resampled data
    resampled_data = []

    # Calculate the maximum size among bins with counts greater than the threshold
    bin_counts = df['deltaG_bin'].value_counts()
    max_size = bin_counts[bin_counts > threshold].max()
    
    # Resample data
    for bin_label in labels:
        bin_data = df[df['deltaG_bin'] == bin_label]
        count = len(bin_data)
        
        if count > threshold:
            # Only oversample if the bin count is above the threshold
            if count < max_size:
                resampled_bin_data = resample(bin_data, replace=True, n_samples=max_size - count, random_state=123)
                resampled_bin_data = pd.concat([bin_data, resampled_bin_data])
            else:
                resampled_bin_data = bin_data
            resampled_data.append(resampled_bin_data)

    # Concatenate all resampled data
    resampled_df = pd.concat(resampled_data)

    # Shuffle the dataframe to mix resampled data
    resampled_df = resampled_df.sample(frac=1).reset_index(drop=True)

    # Set the path where you want to save the file
    output_csv_path = '../../data/ds23_sm_resampled.csv'

    # Save the resampled dataframe as a CSV file
    resampled_df.to_csv(output_csv_path, index=False)