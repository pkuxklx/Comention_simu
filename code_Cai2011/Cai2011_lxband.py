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
# Cai2011Adaptive_Model1, myband
repetition = 100
cv_option = 'pd'
num_cv = 50
folder = 'data'
simu_str = 'lx_band'
cov_str = 'Cai2011Adaptive_Model1'

print(simu_str, cov_str, folder)

for N, T in [(30, 100), (100, 100), (200, 100)] + [(100, 300), (300, 300), (500, 300)]:
    print(N, T)
    if cov_str == 'Cai2011Adaptive_Model1':
        S = gen_S_Cai2011Adaptive_Model1(N = N)
    else:
        raise Exception
    
    R = cov2cor(S)
    
    for eta in [0.1, 0.5, 0.8, 1]:
        print('eta', eta)
        est = []
        for i in range(repetition): 
            X = np.random.RandomState(seed = i).multivariate_normal(mean = np.zeros(N), cov = S, size = T)
            L = gen_L(S, eta, draw_type = 'random', near_factor = None, seed = i)
            c = InfoCorrBand(X, L, num_cv = num_cv)
            R_est, S_est, k = c.auto_fit(cv_option = cv_option, verbose = False)
            est.append((S_est, R_est, k))

            print(i, k)
            
        # save results
        save_dict = {'N': N, 'T': T, 
                     'cov_dscrb': [cov_str], 
                     'simu_dscrb': [simu_str, eta], 
                     'folder': folder, 
                     'is_save': True}
        
        ks = [k for *_, k in est]
        save_data_fig(ks, None, 'k', **save_dict)
        
        for ord in ['fro', 2]:
            print('ord', ord)
            err_cor = [LA.norm(R - R_est, ord) for _, R_est, _ in est]
            err_cov = [LA.norm(S - S_est, ord) for S_est, *_ in est]  

            err_cor = err_cor / LA.norm(R, ord)
            err_cov = err_cov / LA.norm(S, ord)
        
            save_data_fig(err_cor, ord, 'R', **save_dict)
            save_data_fig(err_cov, ord, 'S', **save_dict)

# %%

