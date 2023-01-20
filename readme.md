### Generate Observations

For any $(N, T, S_{N\times N})$, we generate 100 observations as follows. $S$ is the covariance matrix.

```python
import numpy as np
for i in range(100):
    ...
    X = np.random.RandomState(seed = i).multivariate_normal(mean = np.zeros(N), cov = S, size = T)
    L = gen_L(..., rand_seed = i) # only intended for our proposed estimator
    ...
```

### Numerical Simulation

##### Metric

We use the error rate

$$
\frac{\|\hat A \|}{\| A - \hat A \|}
$$

as the metric. Here we investigate the Frobenius norm and the matrix-2 norm.

##### proposed estimator

See [lx_simulation_main.ipynb](lx_simulation_main.ipynb). Results see the folder [data_2023-01-14](data_2023-01-14/).

When $\eta < 1$, $S^L_k$ is a random matrix. It introduces extra randomness to the error rate, and this phenomenon is specific to estimators based on auxiliary information. So for the same $\eta$, I generate $S^L_k$ randomly for $100$ times.

##### other estimators

See [lx_simulation_other.ipynb](lx_simulation_other.ipynb). Results see the file [other_methods.csv](other_methods.csv).

### Data Processing & Query

See [lx_data_process.ipynb](lx_data_process.ipynb).
