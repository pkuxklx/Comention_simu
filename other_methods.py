# %%
import numpy as np, pandas as pd
from utils.adpt_correlation_threshold import AdptCorrThreshold
from utils.covest import NetBanding
from my_api import cov2cor
from wlpy.covariance import Covariance
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

        nb = NetBanding(X, G = np.eye(N), use_correlation = False, num_cv = num_cv)

        R_est, S_est, params = None, None, [None]
        if name == 'Sample':
            S_est = nb.S_sample
            R_est = nb.R_sample
        elif name == 'Soft Threshold' or name == 'Hard Threshold':
            nb.threshold_method = name.lower()
            params = nb.params_by_cv(cv_option = 'pd') # th = params[0]
            S_est = nb.fit(params)
            R_est = cov2cor(S_est)
        
        elif name == 'Linear Shrink':
            S_est = nb.S_lw
            R_est = cov2cor(S_est)
        elif name == 'Nonlinear Shrink':
            assert T > N, f"Nonlinear shrink method is not applicable with T={T} > N={N}."
            S_est = nb.S_nlshrink
            R_est = cov2cor(S_est)
        return R_est, S_est, params