"""Contains functions for evaluating Generative AI Models"""


import pandas as pd
import numpy as np
from typing import List, Tuple


def multivariate_ecdf(data_a: pd.DataFrame,
                      data_b: pd.DataFrame,
                      n_nodes: int = 1000,
                      verbose: bool = True,
                      random_seed: int = None) -> Tuple:
    """
    `multivariate_ecdf(data_a, data_b, n_nodes = 1000, verbose = True, random_seed = None)`

    Function to compute ecdf on proportion of row counts generated
    by dynamic pandas queries on a dataset.
    
    It returns a tuple of query_list and computed ECDFs of both Input Datasets.
    
    The query list is generated using the first dataset as reference. It is a list of query strings combined for each column.
    E.g. Say we have a dataframe with two columns `x0` & `x1`. Then the query string will be generated as `x0 <= z0 and x1 <= z1`, where `z0`, `z1` calculated based on random quantiles that are uniformly distributed on [0, 1], for each feature.
    Many such query_strings are generated and their respective count proportions are calculated for getting the ECDF from both the input datasets.
    
    If we are generating ECDFs for validation/synthetic, validation/training or just real/synthetic, it is very important that the validation or real dataset is the first argument in the function since the query strings will be based on that. The tests will be very effective if run this way. 

    Args:
        data_a (pd.DataFrame): Pandas DataFrame
        data_b (pd.DataFrame): Pandas DataFrame
        n_nodes (int, optional):No of nodes or queries to generate.
                                Defaults to 1000.
        verbose (bool): Flag to display progress of the operations. Defaults to True 
        random_seed (int, optional): random seed to be set before
                                    operations. If set random seed is set using `np.random.seed(random_seed)`. Defaults to None

    Raises:
        TypeError: Throws error if Input Datasets are not Pandas DataFrames

    Returns:
        List: Returns Tuple of query_string & computed ECDFs of both Input Datasets
    """

    if not isinstance(data_a, pd.DataFrame) or not isinstance(data_b, pd.DataFrame):
        raise TypeError("Input Datasets should be Pandas DataFrames!!")

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
    `ks_statistic(ecdf_a, ecdf_b)`
    
    Calculate the KS Statistic between the two input ECDFs.

    Args:
        ecdf_a (List): ECDF Generated through the Multivariate ECDF function
        ecdf_a (List): ECDF Generated through the Multivariate ECDF function

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