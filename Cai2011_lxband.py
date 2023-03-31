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
# %% [markdown]
# # proposed estimator

# %%
# Cai2011Adaptive_Model1, myband
repetition = 100
for N in [100, 300, 500]:
        S = gen_S_Cai2011Adaptive_Model1(N = N)
        cov_str = 'Cai2011Adaptive_Model1'
        # S = gen_S_Cai2011Adaptive_Model2_my(N = N, seed = 0)
        # cov_str = 'Cai2011Adaptive_Model2_my'
        
        R = cov2cor(S)
        for T in [300]:
            print(N, T)
            for ord in ['fro', 2]:
                for eta in [0.5, 0.8, 1]:
                    # nowParam = MyParamsIter(rho, N, T, ord, eta)
                    # lastParam = MyParamsIter(0.5, 500, 100, 2, 0.5)
                    # if nowParam <= lastParam:
                    #     continue

                    err_cor = []
                    err_cov = []            
                    print(ord, eta)
                    
                    for i in range(repetition): 
                        # profiler = Profiler()
                        # profiler.start()
                        
                        X = np.random.RandomState(seed = i).multivariate_normal(mean = np.zeros(N), cov = S, size = T)
                        L = gen_L(S, eta, draw_type = 'random', near_factor = None, seed = i)
                        c = InfoCorrBand(X, L, num_cv = 50)
                        R_est, S_est, k = c.auto_fit(cv_option = 'fast_iter', verbose = False)
                        
                        # # ===== lwg method =====
                        # G = ((c.rowSort <= k) & (c.rowSort.T <= k)).astype(int)
                        # m = NetBanding(X, G)
                        # param_threshold= m.params_by_cv('brute')
                        # S_lwg = m.fit(param_threshold)
                        # # ======================
                        
                        print(i, k)
                        # profiler.stop()
                        # profiler.print()
                        
                        err_cor.append(LA.norm(R - R_est, ord))
                        err_cov.append(LA.norm(S - S_est, ord))
                        # err_lwg.append(LA.norm(S - S_lwg, ord))
                    err_cor = err_cor / LA.norm(R, ord)
                    err_cov = err_cov / LA.norm(S, ord)
                    # err_lwg = err_lwg / LA.norm(S, ord)
                    
                    save_data_fig(err_cor, ord, 'R', N, T, 
                                    cov_dscrb = [cov_str], simu_dscrb = ['lx_band', eta], is_save = 1)
                    save_data_fig(err_cov, ord, 'S', N, T, 
                                    cov_dscrb = [cov_str], simu_dscrb = ['lx_band', eta], is_save = 1)
                    # save_data_fig(err_lwg, ord, 'lwg_cov', draw_type, eta, N, T, rho, near_factor, is_save = 1)

# %%

