# User Guide

## Synthetic vs Real Data Evaluation
- Typically **multivariate_ecdf** & **ks_statistic** functions are used for evaluating real & synthetic tabular data having all **numerical columns**.
- Real data is split into training & validation data. Synthetic data is generated based on the training data and then evaluated with validation.
- While generating ECDFs for validation/synthetic, validation/training or just real/synthetic, it is very important that the validation or real dataset is the first argument in the function since the query strings will be based on that. The tests will be very effective if we follow that approach.
- Higher values of n_nodes can lead to better convergence i.e. smaller difference between **ks_stat** (for validation & synthetic) & **base_ks_stat** (for validation & training). But they tend to get slower since more nodes need to be generated. If ks_stat & base_ks_stat values are similar, increasing the n_nodes to 5000 gives a better performance.
- Absolute Difference i.e. ks_diff = np.abs(ks_stat - base_ks_stat) can be a good measure
- When comparing only real vs synthetic datasets,  `ks_symmterical = np.mean(ks(real,synth),ks(synth,real))` or `ks_symmetrical = np.max(ks(real,synth),ks(synth,real))` can be a good measure

```python
import pandas as pd
import numpy as np
from genai_evaluation import multivariate_ecdf, ks_statistic

random_seed = 42
n_nodes = 5000
verbose = True

# Simulate a real dataset
real_data = pd.DataFrame(np.random.rand(1000, 5), columns=['x1', 'x2', 'x3', 'x4', 'x5'])

#Split real data into 50-50% training & validation dataset
training_data = real_data.sample(frac = 0.5)
validation_data = real_data.drop(training_data.index)

# Simulate a synthetic dataset
synthetic_data = pd.DataFrame(np.random.rand(500, 5), columns=['x1', 'x2', 'x3', 'x4', 'x5'])

# Calculate ECDFs 
_, ecdf_val1, ecdf_synth = \
            multivariate_ecdf(validation_data, 
                              synthetic_data, 
                              n_nodes = n_nodes,
                              verbose = verbose,
                              random_seed=random_seed)

_, ecdf_val2, ecdf_train = \
            multivariate_ecdf(validation_data, 
                              training_data, 
                              n_nodes = n_nodes,
                              verbose = verbose,
                              random_seed=random_seed)

ks_stat = ks_statistic(ecdf_val1, ecdf_synth)
base_ks_stat = ks_statistic(ecdf_val2, ecdf_train)                              

print(f"KS Stat (Synth vs Validation): {ks_stat:.5f}")
print(f"Base KS Stat (Train vs Validation): {base_ks_stat:.5f}")
print(f"Absolute Diff {np.abs((ks_stat-base_ks_stat)**2):.5f}")
```

