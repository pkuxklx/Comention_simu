# -> Created on 03 November 2020
# -> Author: Weiguang Liu
# %% Importing Library
from sklearn import covariance as sk_cov
import nonlinshrink as nls
import numpy as np
import matplotlib.pyplot as plt
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
    
    def hist(self, option = 'off-diag', interval = [0, None], cov = True):
        """
        Plot the histogram of the amplitudes in S_sample or R_sample.
        """
        if interval[0] is None:
            interval[0] = -1
        if interval[1] is None:
            interval[1] = np.inf
        M = self.S_sample if cov else self.R_sample
        M = np.abs(M)
        if option == 'off-diag':
            data = [m for i, row in enumerate(M) for j, m in enumerate(row) if i != j]
        elif option == 'diag':
            data = np.diag(M)
        elif option == 'all':
            data = M.flatten() 
        else:
            raise ValueError('Invalid option.')
        data = [x for x in data if interval[0] <= x <= interval[1]]
        plt.hist(data, bins = self.N)
        plt.title(f"{option}, {'cov' if cov else 'cor'}")
        plt.show()
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