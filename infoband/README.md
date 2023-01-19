## Banding Augmented by Auxiliary Information

### simulation

Given $N, T, S_{N\times N}$, I always generate observations with

```python
rng = np.random.RandomState(seed = 1)
X = rng.multivariate_normal(mean = np.zeros(N), cov = S, size = T)
```

My proposed estimator. See [lx_simulation_main.ipynb](../lx_simulation_main.ipynb)

- $\eta < 1$ introduces extra randomness to the error rate. This phenomenon is specific to estimators based on auxiliary information. 

- Given $\eta$, I generate $S^L_k$ for $100$ times to estimate the average error rate.

Other estimator (linear shrinkage etc.). See [lx_simulation_other.ipynb](../lx_simulation_other.ipynb)

### data processing
Combine the results. See [lx_data_process.ipynb](../lx_data_process.ipynb).