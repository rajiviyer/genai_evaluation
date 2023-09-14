"""Contains functions for evaluating Generative AI Models"""


import pandas as pd
import numpy as np
from typing import List, Tuple
import re


def multivariate_ecdf(data_a: pd.DataFrame,
                      data_b: pd.DataFrame,
                      n_nodes: int = 1000,
                      verbose: bool = True,
                      random_seed: int = None) -> Tuple:
    """
    Args:
        data_a (pd.DataFrame): Pandas DataFrame
        data_b (pd.DataFrame): Pandas DataFrame
        n_nodes (int, optional):Specifies the number of nodes 
                                i.e. the query strings to be generated. 
                                This is a hyperparameter and can be tuned.
                                Defaults to 1000.
        verbose (bool): Flag to display progress of the operations. 
                        Defaults to True 
        random_seed (int, optional): random seed to be set before
                                    operations. If set random seed is set using `np.random.seed(random_seed)`. Defaults to None

    Raises:
        ValueError: Thows value error if one or more column names between both the  
                    Input DataFrames do not match
        TypeError: Throws error if Input Datasets are not Pandas DataFrames

    Returns:
        List: Returns Tuple of query_string & computed ECDFs of both Input Datasets
    """

    if not isinstance(data_a, pd.DataFrame) or not isinstance(data_b, pd.DataFrame):
        raise TypeError("Input Datasets should be Pandas DataFrames!!")
    
    data_a = data_a.copy()
    data_b = data_b.copy()
    data_a_cols = "".join(data_a.columns)
    data_b_cols = "".join(data_b.columns)
    
    if not data_a_cols.isalnum():
        if verbose:
            print("Dataset A columns have special characters. Will be cleaned before processing!!")
        data_a_cols_cleaned = [re.sub(r'[^a-zA-Z0-9]', '', col).lower() 
                               for col in data_a.columns]
        data_a.columns = data_a_cols_cleaned
        
    if not data_b_cols.isalnum():
        if verbose:
            print("Dataset B columns have special characters. Will be cleaned before processing!!")
        data_b_cols_cleaned = [re.sub(r'[^a-zA-Z0-9]', '', col).lower() 
                               for col in data_b.columns]
        data_b.columns = data_b_cols_cleaned
    
    if not data_a.columns.equals(data_b.columns):
        raise ValueError("One or more column names do not match in both DataFrames!!")    

    eps = 0.0000000001
    query_val = []
    features = data_a.columns
    n_features = len(features)
    if random_seed:
        np.random.seed(random_seed)
    
    for point in range(n_nodes):
        if point % 100 == 0 and verbose:
            print(f"Sampling ecdf, location = {point}")
        
        # Get random percentiles
        percentiles = np.random.uniform(0, 1, n_features)
        percentiles = percentiles**(1/n_features)

        # Get the percentile values from the dataset a for each column
        perc_vals = [eps + np.quantile(data_a.iloc[:, k], perc)
                     for k, perc in enumerate(percentiles)]

        # Create the query string combined for each column
        query_str = " and ".join([f"{features[k]} <= {perc_val}"
                                  for k, perc_val in enumerate(perc_vals)])

        # From dataset a, get the counts of rows which
        # satisfy the conditions in the query string
        filter_count_a = len(data_a.query(query_str))

        # For counts > 0, create key: str of the list of perc_vals
        # Append key, query_str & the normalized filter count for dataset
        if filter_count_a > 0:
            key = ', '.join(map(str, perc_vals))
            query_val.append((key, query_str, filter_count_a/len(data_a)))

    # Sort the list based on the items (third element of each tuple)
    query_val.sort(key=lambda item: item[2])

    query_lst = []
    ecdf_a = []
    ecdf_b = []

    # for each entry in the query_val list
    # Retrieve the query_str and run on both datasets to get the filter counts
    # Normalize the filter count for dataset b

    for e_val in query_val:
        query_str = e_val[1]
        value_data_a = e_val[2]
        filter_count_b = len(data_b.query(query_str))
        value_data_b = filter_count_b / len(data_b)
        query_lst.append(query_str)
        ecdf_a.append(value_data_a)
        ecdf_b.append(value_data_b)

    return query_lst, ecdf_a, ecdf_b


def ks_statistic(ecdf_a: List, ecdf_b: List) -> float:
    """
    Args:
        ecdf_a (List): ECDF Generated through the Multivariate ECDF function
        ecdf_a (List): ECDF Generated through the Multivariate ECDF function

    Raises:
        ValueEror: Throws error if the input ECDFs' are empty or their lengths are not equal

    Returns:
        float: Returns KS Statistic
    """
    if len(ecdf_a) != len(ecdf_b):
        raise ValueError("Both Input ECDFs should be of the same length!!")

    if len(ecdf_a) == 0 or len(ecdf_b) == 0:
        raise ValueError("ECDFs should not be empty!!")

    np_ecdf_a = np.array(ecdf_a)
    np_ecdf_b = np.array(ecdf_b)
    
    # Compute the absolute difference between ECDFs
    abs_diff = np.abs(np_ecdf_a - np_ecdf_b)

    # Find the maximum absolute difference (KS statistic)
    ks_statistic = np.max(abs_diff)

    return ks_statistic
