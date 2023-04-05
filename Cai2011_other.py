# %%
import numpy as np
from sklearn.datasets import make_sparse_spd_matrix
from scipy import linalg as LA
import pandas as pd
import time, os
import matplotlib.pyplot as plt

# from pyinstrument import Profiler

from infoband.band_info import InfoCorrBand
from wlpy.covariance import Covariance
from utils.adpt_correlation_threshold import AdptCorrThreshold
from wlpy.gist import heatmap
from utils.covest import NetBanding

import warnings
warnings.filterwarnings("ignore")

from my_api import *
# %%
class Other_Methods():
    def __init__(self):
        self.names = ['Sample', 'Soft Threshold', 'Hard Threshold', 'Linear Shrink', 'Nonlinear Shrink']

    def fit(self, name, X: np.ndarray):
        """
        Args:
            name: A method's name.
        """
        assert name in self.names, "Invalid method name."
        T, N = X.shape
        G_zero = np.ones((N, N)) - np.eye(N)
        m = AdptCorrThreshold(pd.DataFrame(X), G_zero)
        if name == 'Sample':
            S_est = m.sample_cov()
            R_est = cov2cor(S_est)
        elif name == 'Soft Threshold':
            R_est, S_est, _ = m.auto_fit(threshold_method = 'soft threshold')
        elif name == 'Hard Threshold':
            R_est, S_est, _ = m.auto_fit(threshold_method = 'hard threshold')
        elif name == 'Linear Shrink':
            S_est = m.lw_lin_shrink()
            R_est = cov2cor(S_est)
        elif name == 'Nonlinear Shrink':
            assert T > N, f"Nonlinear shrink method is not applicable with T={T} > N={N}."
            S_est = m.nonlin_shrink()
            R_est = cov2cor(S_est)
        return R_est, S_est
# %%
# Cai2011Adaptive, other methods
repetition = 100
om = Other_Methods() 

for N in [100, 300, 500]:
    S = gen_S_Cai2011Adaptive_Model1(N = N)
    cov_str = 'Cai2011Adaptive_Model1'
    # S = gen_S_Cai2011Adaptive_Model2_my(N = N, seed = 0)
    # cov_str = 'Cai2011Adaptive_Model2_my'
    
    R = cov2cor(S)
    for T in [300]:
        print(N, T)
        for ord in ['fro', 2]:
            for j, method_name in enumerate(om.names):
                nowParam = MyParamsIter(N, T, ord, j)
                lastParam = MyParamsIter(300, 300, 'fro', 3)
                if nowParam <= lastParam:
                    continue

                if method_name == 'Nonlinear Shrink' and N >= T:
                    continue

                err_cor = []
                err_cov = []            
                print(ord, method_name)
                
                for i in range(repetition): 
                    # profiler = Profiler()
                    # profiler.start()
                    
                    X = np.random.RandomState(seed = i).multivariate_normal(mean = np.zeros(N), cov = S, size = T)
                    
                    R_est, S_est = om.fit(method_name, X)

                    print(i, end = ' ')
                    # profiler.stop()
                    # profiler.print()
                    
                    err_cor.append(LA.norm(R - R_est, ord))
                    err_cov.append(LA.norm(S - S_est, ord))
                    
                err_cor = err_cor / LA.norm(R, ord)
                err_cov = err_cov / LA.norm(S, ord)
                
                cov_dscrb = [cov_str]
                simu_dscrb = [method_name]
                save_data_fig(err_cor, ord, 'R', N, T, 
                                cov_dscrb = cov_dscrb, simu_dscrb = simu_dscrb, is_save = 1)
                save_data_fig(err_cov, ord, 'S', N, T, 
                                cov_dscrb = cov_dscrb, simu_dscrb = simu_dscrb, is_save = 1)
# %%

