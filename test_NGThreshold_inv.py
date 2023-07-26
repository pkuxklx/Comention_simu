# %%
import numpy as np
import pandas as pd
# from infoband.band_info import InfoCorrBand
from utils.covest import NetBanding
from wlpy.gist import heatmap

from my_api import *
from sklearn.datasets import make_spd_matrix, make_sparse_spd_matrix
# %%
N, T = 500, 100
S = make_sparse_spd_matrix(dim = N, alpha = 0.95, random_state = 1)
# S = gen_S_Cai2011Adaptive_Model2_my(N = N, seed = 0, probB = 10 / (N // 2))
heatmap(S)
# %%
X = np.random.RandomState(seed = 1).multivariate_normal(mean = np.zeros(N), cov = S, size = T)
# %%
L = np.eye(N)
th_m = 'soft threshold'
m = NetBanding(X, G = L, use_correlation = True, num_cv = 50, threshold_method = th_m)
# %%
params = m.params_by_cv_inv(cv_option = 'grid')
# %%
th, log10_eps = params
eps = 10 ** log10_eps
S_est = m.fit([th], ad_option = 'pd', eps = eps)
print('th', th)
print('smallest_eig', np.linalg.eigvalsh(S_est).min())
print('loss_inv', np.linalg.norm(np.linalg.inv(S) - np.linalg.inv(S_est)))
# %%

# %%
from scipy import optimize
import numpy as np

def f(x):
    return np.linalg.norm(x - np.array([3, 2]))

optimize.brute(f, 
            (slice(0, 4, 0.11), slice(0, 4, 0.5)))
# %%
