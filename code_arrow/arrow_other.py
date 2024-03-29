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

from infoband.band_info import InfoCorrBand
from wlpy.covariance import Covariance
from utils.adpt_correlation_threshold import AdptCorrThreshold
from wlpy.gist import heatmap
from utils.covest import NetBanding

import warnings
warnings.filterwarnings("ignore")

from my_api import *
# %%
from other_methods import Other_Methods
# %%
# arrow, other methods
repetition = 100
om = Other_Methods() 
folder = 'data_arrow'
cov_str = 'arrow'

print('other methods', cov_str, folder)

for N in [100, 300, 500]:
    S = gen_S_arrow(N = N)
    
    R = cov2cor(S)
    for T in [300]:
        print(N, T)
        for ord in ['fro', 2]:
            for j, method_name in enumerate(om.names):
                # nowParam = MyParamsIter(N, T, ord, j)
                # lastParam = MyParamsIter(300, 300, 'fro', 3)
                # if nowParam <= lastParam:
                #     continue
                # if method_name not in ['Soft Threshold', 'Hard Threshold']:
                #     continue

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
                                cov_dscrb = cov_dscrb, simu_dscrb = simu_dscrb, is_save = 1, folder = folder)
                save_data_fig(err_cov, ord, 'S', N, T, 
                                cov_dscrb = cov_dscrb, simu_dscrb = simu_dscrb, is_save = 1, folder = folder)