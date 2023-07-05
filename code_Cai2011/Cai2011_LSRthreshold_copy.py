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
repetition = 100
cv_option = 'brute'
cmap = 'gist_gray_r'
num_cv = 50
folder = 'data'
simu_str = 'LSRthreshold'
cov_str = 'Cai2011Adaptive_Model2_my'

print(simu_str, cov_str, folder)

for N, T in [(30, 100), (100, 100), (200, 100)] + [(100, 300), (300, 300), (500, 300)]:
    print(N, T)
    if cov_str == 'Cai2011Adaptive_Model2_my':
        S = gen_S_Cai2011Adaptive_Model2_my(N = N, seed = 0, probB = 10 / (N // 2))
    else:
        raise Exception
    
    R = cov2cor(S)

    for l in np.linspace(0, 1, 6):
        for prob in [0.5, 0.8, 1]:
            for qrob in [0, 0.1, 0.2]:
                if (N,T,l,prob,qrob) != (200,100,0,1,0):
                    continue

                print('probability', l, prob, qrob)
                est = []
                for i in range(repetition): 
                    # profiler = Profiler()
                    # profiler.start()
                    
                    X = np.random.RandomState(seed = i).multivariate_normal(mean = np.zeros(N), cov = S, size = T)

                    from utils.simulation import func_G
                    hatL = func_G(S, prob, qrob, observe_level = l, seed = i) # binary auxiliary set

                    m = NetBanding(X, G = hatL, use_correlation = False, num_cv = num_cv)
                    param_threshold = m.params_by_cv(cv_option = cv_option)
                    th = param_threshold[0]
                    S_est = m.fit(param_threshold)
                    R_est = cov2cor(S_est)
                    est.append((S_est, R_est, th))

                    ord = 'fro' # tmp
                    print(i, end = ' ')
                    normS = LA.norm(S, ord)
                    normErr = LA.norm(S - S_est, ord)
                    print(f'{i}, param_threshold: {th :.2f}, rela: {normErr / normS :.2f}, abs: {normErr :.2f}', end = '\n')

                # save results
                simu_params = [l, prob, qrob, cv_option, num_cv]
                save_dict = {'N': N, 'T': T, 
                                'cov_dscrb': [cov_str], 
                                'simu_dscrb': [simu_str, *simu_params], 
                                'folder': folder, 
                                'is_save': True}
                
                ths = [th for *_, th in est]
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

# %%
