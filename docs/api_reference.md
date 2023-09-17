# API Reference

**multivariate_ecdf**(*data_a, data_b, n_nodes = 1000, verbose = True, random_seed = None*)


Function to compute ecdf on proportion of row counts generated
by dynamic pandas queries on a dataset.

It returns a tuple of query_list and computed ECDFs of both Input Datasets.

The query list is generated using the first dataset as reference. It is a list of query strings combined for each column.
E.g. Say we have a dataframe with two columns **`x0`** & **`x1`**. Then the query string will be generated as **`x0 <= z0 and x1 <= z1`**, where **`z0`**, **`z1`** calculated based on random quantiles that are uniformly distributed on [0, 1], for each feature.
Many such query_strings are generated and their respective count proportions are calculated for getting the ECDF from both the input datasets.

::: genai_evaluation.multivariate_ecdf
--------------------------------
**ks_statistic**(*ecdf_a, ecdf_b*)

Function to calculate the KS Statistic between the two input ECDFs.
Calculates the maximum separation (distance) between the ECDFs and yields a result ranging from 0 (indicating the best fit) and 1 (indicating the worst fit).

::: genai_evaluation.ks_statistic