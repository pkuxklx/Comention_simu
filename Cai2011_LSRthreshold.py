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
# Cai2011Adaptive_Model1, and Model2_my ; LSRthreshold, use_correlation = False
repetition = 100
cv_option = 'brute'
cmap = 'gist_gray_r'

for N in [100, 300, 500]:
        simu_str = 'LSRthreshold'
        S = gen_S_Cai2011Adaptive_Model1(N = N)
        cov_str = 'Cai2011Adaptive_Model1'
        # S = gen_S_Cai2011Adaptive_Model2_my(N = N, seed = 0)
        # cov_str = 'Cai2011Adaptive_Model2_my'
        # heatmap(S, cmap = cmap)

        R = cov2cor(S)
        for T in [300]:
            print(N, T)
            for ord in ['fro', 2]:
                for tau in [0.2]:
                    for prob in [0.9, 0.99]:
                        for qrob in [0.01, 0.1]:
                            nowParam = MyParamsIter(N, T, ord, tau, prob, qrob)
                            # lastParam = MyParamsIter(0.5, 500, 100, 2, 0.5)
                            # if nowParam <= lastParam:
                            #     continue

                            if nowParam > MyParamsIter(100, 300, 'fro', 0.2, 0.99, 0.01):
                                break
                            

                            err_cor = []
                            err_cov = []            
                            print(ord, tau, prob, qrob)
                            
                            for i in range(repetition): 
                                # profiler = Profiler()
                                # profiler.start()
                                
                                X = np.random.RandomState(seed = i).multivariate_normal(mean = np.zeros(N), cov = S, size = T)

                                from utils.simulation import func_G
                                hatL = func_G(S, prob, qrob, observe_level = tau, seed = i) # binary auxiliary set

                                m = NetBanding(X, G = hatL, use_correlation = False)
                                param_threshold = m.params_by_cv(cv_option = cv_option)
                                S_est = m.fit(param_threshold)
                                R_est = cov2cor(S_est)

                                print(i)
                                # heatmap(S_est, cmap = cmap)
                                # profiler.stop()
                                # profiler.print()
                                
                                err_cor.append(LA.norm(R - R_est, ord))
                                err_cov.append(LA.norm(S - S_est, ord))
                            
                            err_cor = err_cor / LA.norm(R, ord)
                            err_cov = err_cov / LA.norm(S, ord)
                            
                            simu_params = [tau, prob, qrob, cv_option]
                            save_data_fig(err_cor, ord, 'R', N, T, 
                                            cov_dscrb = [cov_str], simu_dscrb = [simu_str, *simu_params, 'brute'], is_save = 1)
                            save_data_fig(err_cov, ord, 'S', N, T, 
                                            cov_dscrb = [cov_str], simu_dscrb = [simu_str, *simu_params, 'brute'], is_save = 1)

# %%

# %%
