# %%
import sys, os
os.chdir('..')
sys.path.append(os.getcwd())
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
# Cai2011Adaptive, other methods
from other_methods import Other_Methods
om = Other_Methods() 
repetition = 100
folder = 'data'
cov_str = 'Cai2011Adaptive_Model2_my'

for N, T in [(30, 100), (100, 100), (200, 100)] + [(100, 300), (300, 300), (500, 300)]:
    print(N, T)
    if cov_str == 'Cai2011Adaptive_Model2_my':
        S = gen_S_Cai2011Adaptive_Model2_my(N = N, seed = 0, probB = 10 / (N // 2))
    else:
        raise Exception
    
    R = cov2cor(S)

    for j, method_name in enumerate(om.names):
        if method_name == 'Nonlinear Shrink' and N >= T:
            continue

        print(method_name)
        est = []
        for i in range(repetition): 
            X = np.random.RandomState(seed = i).multivariate_normal(mean = np.zeros(N), cov = S, size = T)
            R_est, S_est, params = om.fit(method_name, X)
            est.append((S_est, R_est, params))

            print(i, end = ' ')
        
        # save results
        save_dict = {'N': N, 'T': T, 
                    'cov_dscrb': [cov_str], 
                    'simu_dscrb': [method_name], 
                    'folder': folder, 
                    'is_save': True}
        
        if 'Threshold' in method_name:
            ths = [params[0] for *_, params in est]
            save_data_fig(ths, None, 'th', **save_dict)
            
        for ord in ['fro', 2]:
            print('ord', ord)
            err_cor = [LA.norm(R - R_est, ord) for _, R_est, _ in est]
            err_cov = [LA.norm(S - S_est, ord) for S_est, *_ in est] 

            err_cor = err_cor / LA.norm(R, ord)
            err_cov = err_cov / LA.norm(S, ord)
        
            save_data_fig(err_cor, ord, 'R', **save_dict)
            save_data_fig(err_cov, ord, 'S', **save_dict)
# %%

