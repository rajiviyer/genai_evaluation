# Home
[![PyPI version](https://badge.fury.io/py/genai-evaluation.svg)](https://badge.fury.io/py/genai-evaluation)

GenAI Evaluation is a library which contains methods to evaluate differences in Real & Synthetic Data. 

## Functions
- **multivariate_ecdf**: Computes joint or multivariate ECDF in contrast to the univariate capabilities provided by packages like statsmodels
- **ks_statistic**: Calculates the KS Statistic for two multivariate ECDFs  

Read more in the [API Reference](./api_reference.md) & [User Guide](./user_guide.md) pages.

## Authors
- [Dr. Vincent Granville](mailto:vincentg@mltechniques.com) - Research
- [Rajiv Iyer](mailto:raju.rgi@gmail.com) - Development/Maintenance

## Installation
The package can be installed with
```
pip install genai_evaluation
```

## Tests
The test can be run by cloning the repo and running:
```
pytest tests
```
In case of any issues running the tests, please run them after installing the package locally:

```
pip install -e .
```

## Usage

Start by importing the class
```Python
from genai_evaluation import multivariate_ecdf, ks_statistic
```

Assuming we have two pandas dataframes (Real & Synthetic) and only numerical columns, we pass them to the multivariate_ecdf function which returns the computed multivariate ECDFs of both.
```Python
query_str, ecdf_real, ecdf_synth = multivariate_ecdf(real_data, synthetic_data, n_nodes = 1000, verbose = True)
```

We then calculate the multivariate KS Distance between the ECDFs
```Python
ks_stat = ks_statistic(ecdf_real, ecdf_synth)
```

## Motivation
The motivation for this package comes from Dr. Vincent Granville's paper [Generative AI Technology Break-through: Spectacular Performance of New Synthesizer](https://mltechniques.com/2023/08/02/generative-ai-technology-break-through-spectacular-performance-of-new-synthesizer/)

If you have any tips or suggestions, please contact us on email.