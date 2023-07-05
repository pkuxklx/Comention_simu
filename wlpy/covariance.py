# -> Created on 03 November 2020
# -> Author: Weiguang Liu
# %% Importing Library
from sklearn import covariance as sk_cov
import nonlinshrink as nls
import numpy as np
# %%


class Covariance():
    def __init__(self, X: np.array):
        """
        Covariance estimation from the observations.
        Args:
            X: T * N
        """
        self.X = X
        self.T, self.N = X.shape

    @property
    def S_sample(self):
        return np.cov(np.array(self.X), rowvar=False)

    @property
    def R_sample(self):
        return np.corrcoef(np.array(self.X), rowvar= False)
    
    @property
    def D_sample(self):
        return np.diag(np.diag(self.S_sample)) ** 0.5

    def lw_lin_shrink(self):
        """
        lw stands for Ledoit and Wolf 2004
        """
        S_lw = sk_cov.LedoitWolf().fit(self.X).covariance_
        return S_lw
    
    @property
    def S_lw(self):
        return self.lw_lin_shrink()

    def nonlin_shrink(self):
        S_nlshrink = nls.shrink_cov(self.X)
        return S_nlshrink
    
    @property
    def S_nlshrink(self):
        return self.nonlin_shrink()
    
    '''
    def network_hard_threshold(self, G, est_cov=None):
        """
        This calculates the hard-threshold estimator using a network matrix G. 
        The original estimator is est_cov
        """
        if est_cov == None:
            _S = self.sample_cov()
        else:
            _S = est_cov
        # threshold by network: if G_ij = 0, est_cov_ij = 0 
        _S[np.where((G + np.eye(self.N)) == 0)] = 0
        return _S
    '''
# %%  Heatmap of the cov matrix S


# -> Xiang Lu, on 2022/10/2
# %% test
G = np.array([
    [0, 1, 0], 
    [1, 0, 0], 
    [0, 0, 0]
])
Sigma = np.array([
    [2, 1, 0.2], 
    [1, 3, -0.9], 
    [0.2, -0.9, 1.5]
])
X1 = np.random.multivariate_normal(mean = np.zeros(3), 
                                   cov = Sigma, 
                                   size = 50)
# %%
# c = Covariance(X1)
# c.network_hard_threshold(G)
# %%
# c.sample_cov()