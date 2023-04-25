# %%
import numpy as np, pandas as pd
from utils.adpt_correlation_threshold import AdptCorrThreshold
from utils.covest import NetBanding
from my_api import cov2cor
# %%
class Other_Methods():
    def __init__(self):
        self.names = ['Sample', 'Soft Threshold', 'Hard Threshold', 'Linear Shrink', 'Nonlinear Shrink']

    def fit(self, name, X: np.ndarray, num_cv = 50):
        """
        Args:
            name: A method's name.
            num_cv: Used only when name is 'Soft Threshold' or 'Hard Threshold'.
        """
        assert name in self.names, "Invalid method name."
        T, N = X.shape
        G_zero = np.ones((N, N)) - np.eye(N)
        m = AdptCorrThreshold(pd.DataFrame(X), G_zero)


        if name == 'Sample':
            S_est = m.sample_cov()
            R_est = cov2cor(S_est)
        elif name == 'Soft Threshold' or name == 'Hard Threshold':
            nb = NetBanding(X, G = np.zeros((N, N)), use_correlation = False, num_cv = num_cv, threshold_method = name.lower())
            tau = nb.params_by_cv(cv_option = 'brute')
            S_est = nb.fit(tau)
            R_est = cov2cor(S_est)
        
        elif name == 'Linear Shrink':
            S_est = m.lw_lin_shrink()
            R_est = cov2cor(S_est)
        elif name == 'Nonlinear Shrink':
            assert T > N, f"Nonlinear shrink method is not applicable with T={T} > N={N}."
            S_est = m.nonlin_shrink()
            R_est = cov2cor(S_est)
        return R_est, S_est