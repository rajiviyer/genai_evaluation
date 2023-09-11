# GENAI EVALUATION
GenAI Evaluation is a library which contains methods to evaluate differences in Real & Synthetic Data. 

# Authors
- [Dr. Vincent Granville](mailto:vincentg@mltechniques.com)
- [Rajiv Iyer](mailto:raju.rgi@gmail.com)

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
If this does not work, the package might not currently be findable. In that case, please install it locally with:

```
pip install -e .
```

## Usage

Start by importing the class
```Python
from genai_evaluation import multivariate_ecdf, ks_distance
```

Assuming we have two pandas dataframes (Real & Synthetic) and only numerical columns, we pass them to the multivariate_ecdf function which returns the computed multivariate ECDFs of both.
```Python
ecdf_real, ecdf_synth = multivariate_ecdf(real_data, synthetic_data, n_nodes = 1000, verbose = True)
```

We then calculate the multivariate KS Distance between the ECDFs
```Python
ks_stat = ks_distance(ecdf_real, ecdf_synth)
```

## Motivation
The motivation for this package comes from Dr. Vincent Granville's book [Generative AI Technology Break-through: Spectacular Performance of New Synthesizer](https://mltblog.com/3Koag20)

If you have any tips or suggestions, please contact us on email.