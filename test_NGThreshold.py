# %%
import numpy as np
import pandas as pd
# from infoband.band_info import InfoCorrBand
from utils.covest import NetBanding
from wlpy.gist import heatmap

from my_api import *
from sklearn.datasets import make_spd_matrix, make_sparse_spd_matrix
# %%
N, T = 200, 100
# S = make_sparse_spd_matrix(dim = N, alpha = 0.05, random_state = 1)
S = gen_S_Cai2011Adaptive_Model2_my(N = N, seed = 0, probB = 10 / (N // 2))
heatmap(S)
# %%
X = np.random.RandomState(seed = 1).multivariate_normal(mean = np.zeros(N), cov = S, size = T)
# %%
L2 = (np.abs(S)>0)
m = NetBanding(X, G = L2, use_correlation = False, num_cv = 10)
params = m.params_by_cv(cv_option = 'brute')
th = params[0]
print('th', th, th * m.scaling_factor)
S_est = m.fit(params, ad_option = 'pd')
print('smallest_eig', np.linalg.eigvalsh(S_est).min())
print('loss', np.linalg.norm(S - S_est))
# %%
def get_abs_undiag_max(S):
    np.fill_diagonal(S, 0)    
    return np.abs(S).max()

assert (S_est == S_est.T).all()
# %%
from scipy import optimize
result = optimize.brute(
    m.loss_func, (slice(0, 36, 0.1
                        ),), full_output = True)

import matplotlib.pyplot as plt
plt.plot(result[2], result[3])
plt.show()
# %%
vals = []
for param in result[2]:
    S_est = m.fit(param)
    vals.append(np.linalg.eigvals(S_est).min())
plt.plot(result[2], vals)
plt.show()
# %%


# %%
def f1(x):
    return (x-1)**2
from scipy import optimize
optimize.brute(f1, ((3, 10), ), full_output = False)

# optimize.minimize(
    # f1, np.array([3]), method = 'trust-constr', bounds = ((2, None),)).x
# %%
import scipy.optimize as opt

def objective(x):
    return x[0]**2 + x[1]**2  # Example objective function

def constraint(x):
    return x[0] + x[1] - 1  # Example constraint: x[0] + x[1] - 1 >= 0

x0 = [0, 0]  # Initial guess

con = {'type': 'ineq', 'fun': constraint}

bounds = ((-1, 1), (-1, 1))

res = opt.minimize(objective, x0, bounds=bounds, constraints=con)

print(res.x) 

# %%
import scipy.optimize as opt

def objective(x):
    return x[0]**2 + x[1]**2  # Example objective function

def constraint(x):
    return x[0] + x[1] - 1  # Example constraint: x[0] + x[1] - 1 >= 0

x0 = [0, 0]  # Initial guess

# Define the constraints
con = {'type': 'ineq', 'fun': constraint}

# Define the optimization bounds (if needed)
bounds = ((-1, 1), (-1, 1))

# Run the optimization using 'trust-constr' method
res = opt.minimize(objective, x0, method='trust-constr', bounds=bounds, constraints=con)

print(res.x)  # Optimized solution

# %%
