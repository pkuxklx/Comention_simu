# %%
import sys, os
os.chdir('..')
sys.path.append(os.getcwd())
# %%
import numpy as np
from sklearn.datasets import make_sparse_spd_matrix
from scipy import linalg as LA
import pandas as pd
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
# Cai2011Adaptive_Model1, and Model2_my ; LSRthreshold, use_correlation = False
repetition = 20
cv_option = 'brute'
cmap = 'gist_gray_r'
num_cv = 50
folder = 'data'
simu_str = 'LSRthreshold'
cov_str = 'Cai2011Adaptive_Model2_my'

print(simu_str, cov_str, folder)

for N in [300, 100, 500]:
    if cov_str == 'Cai2011Adaptive_Model1':
        S = gen_S_Cai2011Adaptive_Model1(N = N)
    elif cov_str == 'Cai2011Adaptive_Model2_my':
        S = gen_S_Cai2011Adaptive_Model2_my(N = N, seed = 0)
    else:
        raise Exception
    
    R = cov2cor(S)
    for T in [300]:
        print(N, T)
        
        for tau in [0.2]:
            for prob in [1]: # [0.9, 0.99, 1]:
                for qrob in [0]: # [0, 0.01, 0.1]:
                    for ord in ['fro', 2]:
                        # nowParam = MyParamsIter(N, T, ord)
                        # lastParam = MyParamsIter(100, 300, 'fro')
                        # if nowParam != lastParam:
                        #     continue

                        # if not (prob == 1 and qrob == 0 and N == 500):
                        #     continue

                        err_cor = []
                        err_cov = []            
                        print(ord, tau, prob, qrob)
                        
                        for i in range(repetition): 
                            # profiler = Profiler()
                            # profiler.start()
                            
                            X = np.random.RandomState(seed = i).multivariate_normal(mean = np.zeros(N), cov = S, size = T)

                            from utils.simulation import func_G
                            hatL = func_G(S, prob, qrob, observe_level = tau, seed = i) # binary auxiliary set

                            m = NetBanding(X, G = hatL, use_correlation = False, num_cv = num_cv)
                            param_threshold = m.params_by_cv(cv_option = cv_option)
                            S_est = m.fit(param_threshold)
                            R_est = cov2cor(S_est)

                            # print(i, end = ' ')
                            normS = LA.norm(S, ord)
                            normErr = LA.norm(S - S_est, ord)
                            print(f'{i}, param_threshold: {param_threshold[0] :.2f}, rela: {normErr / normS :.2f}, abs: {normErr :.2f}', end = '\n')
                            
                            err_cor.append(LA.norm(R - R_est, ord))
                            err_cov.append(LA.norm(S - S_est, ord))

                        err_cor = err_cor / LA.norm(R, ord)
                        err_cov = err_cov / LA.norm(S, ord)
                        
                        simu_params = [tau, prob, qrob, cv_option, num_cv]
                        save_data_fig(err_cor, ord, 'R', N, T, 
                                        cov_dscrb = [cov_str], simu_dscrb = [simu_str, *simu_params], is_save = 1, folder = folder)
                        save_data_fig(err_cov, ord, 'S', N, T, 
                                        cov_dscrb = [cov_str], simu_dscrb = [simu_str, *simu_params], is_save = 1, folder = folder)
# %%

# %%
