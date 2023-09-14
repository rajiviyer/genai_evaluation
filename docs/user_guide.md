# User Guide

## Synthetic vs Real Data Evaluation
- Typically **multivariate_ecdf** & **ks_statistic** functions are used for evaluating real & synthetic tabular data having all **numerical columns**.
- Real data is split into training & validation data. Synthetic data is generated based on the training data and then evaluated with validation.
- When calling the ECDF function, the first argument must be the real data used to produce the synthetization, and the second argument the synthesized data . In a cross-validation setting, the first argument should be the validation data: a portion of the real data called holdout, and not used to train the synthesizer. The reason for this is that the real or validation set is used as the reference dataset. The goal is to test whether the distribution observed in the synthetization matches that in the real or validation set, not the other way around.
- The ECDF is approximated using a number of locations in the feature space, specified by the argument n_nodes. The larger n_nodes, the better the approximation, leading to a more accurate KS_statistic. This is especially important in cross-validation when comparing the KS distance between the validation set and synthetized data, with KS_Base measuring the distance between the validation and training sets. If both distances are very similar, indicating a good synthetization, you may increase n_nodes to get a more refined evaluation. Increasing n_nodes results in more computing time.
- For a symmetrical distance, use KS_symmetrical = max(KS_1, KS_2) with KS_1 = KS(validation, synthetic) and KS_2 = KS(synthetic, validation).You may replace validation by real or training data, if you do not perform cross-validation.  

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
print(f"Absolute Diff {np.abs((ks_stat-base_ks_stat)):.5f}")
```

