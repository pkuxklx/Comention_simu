# %%
import os, sys
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
# arrow, myband
repetition = 100
cv_option = 'fast_iter'
num_cv = 50
folder = 'data_arrow'
simu_str = 'lx_band'
cov_str = 'arrow'

print(simu_str, cov_str, folder)

for N in [100, 300, 500]:
    if cov_str == 'arrow':
        S = gen_S_arrow(N = N)
    else:
        raise Exception('Invalid cov_str.')

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
                    X = np.random.RandomState(seed = i).multivariate_normal(mean = np.zeros(N), cov = S, size = T)
                    L = gen_L(S, eta, draw_type = 'random', near_factor = None, seed = i)
                    c = InfoCorrBand(X, L, num_cv = num_cv)
                    R_est, S_est, k = c.auto_fit(cv_option = cv_option, verbose = False)
                    print(i, k)
                    
                    heatmap((S != 0));
                    heatmap((S_est != 0));

                    err_cor.append(LA.norm(R - R_est, ord))
                    err_cov.append(LA.norm(S - S_est, ord))
                    
                err_cor = err_cor / LA.norm(R, ord)
                err_cov = err_cov / LA.norm(S, ord)
               
                
                save_data_fig(err_cor, ord, 'R', N, T, 
                                cov_dscrb = [cov_str], simu_dscrb = ['lx_band', eta], is_save = 1, folder = folder)
                save_data_fig(err_cov, ord, 'S', N, T, 
                                cov_dscrb = [cov_str], simu_dscrb = ['lx_band', eta], is_save = 1, folder = folder)

# %%

