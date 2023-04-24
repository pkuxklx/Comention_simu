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
repetition = 30
N = 100
T = 300
ord = 'fro'
S = gen_S_Cai2011Adaptive_Model2_my(N = N, seed = 0)
ths = []
for i in range(repetition):
    print(i)
    X = np.random.RandomState(seed = i).multivariate_normal(mean = np.zeros(N), cov = S, size = T)
    nb = NetBanding(X, G = np.zeros((N, N)), use_correlation = False, num_cv = 50, threshold_method = 'hard threshold')
    th = nb.params_by_cv(cv_option = 'brute')[0]
    ths.append(th)
# %%
np.savetxt(fname = 'data/Cai2011Model2_hardthres_th.txt', X = ths)

# %%
