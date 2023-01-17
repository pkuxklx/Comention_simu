### Generate Observations
Set the seed as follows for any $(N, T, S_{N\times N})$. $S$ is the covariance matrix.
```python
import numpy as np
rng = np.random.RandomState(seed = 1)
X = rng.multivariate_normal(mean = np.zeros(N), cov = S, size = T)
```

### Simulation
##### Metric

We use the error rate 
\[ \frac{\|\hat A \|}{\| A - \hat A \|} \]

as the metric. Here we investigate the Frobenius norm and the matrix-2 norm.

##### proposed estimator
See [lx_simulation_main.ipynb](lx_simulation_main.ipynb).

When $\eta < 1$, $S^L_k$ is a random matrix. It introduces extra randomness to the error rate, and this phenomenon is specific to estimators based on auxiliary information. So for the same $\eta$, I generate $S^L_k$ randomly for $100$ times.


##### other estimators
See [lx_simulation_other.ipynb](lx_simulation_other.ipynb).

### Data Processing
