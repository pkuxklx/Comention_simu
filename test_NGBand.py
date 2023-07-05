# %%
import numpy as np
import pandas as pd
# from infoband.band_info import InfoCorrBand
from utils.covest import NetBanding
from infoband.band_info import InfoCorrBand
from wlpy.gist import heatmap

from my_api import *
from sklearn.datasets import make_spd_matrix, make_sparse_spd_matrix
# %%
N, T = 200, 100
alpha = 0.9
S = make_sparse_spd_matrix(dim = N, alpha = alpha, random_state = 1)
heatmap(S)
# %%
X = np.random.RandomState(seed = 1).multivariate_normal(mean = np.zeros(N), cov = S, size = T)
# %%
L = gen_L(S, eta = 1, draw_type = 'random', near_factor = None, seed = 1)
            
c = InfoCorrBand(X, L, num_cv = 10)

params = c.params_by_cv(cv_option = 'fast_iter') 
S_est = c.fit(params, ad_option = 'pd')       
k = params[0]
R_est = cov2cor(S_est)

print('k', k)
print('smallest_eig', np.linalg.eigvalsh(S_est).min())
print('loss', np.linalg.norm(S - S_est))
# %%
vals = []
for k in range(1, N + 1):
    params = np.array([k])
    vals.append(np.linalg.eigvalsh(c.fit(params)).min())
plt.plot(range(1, N + 1), vals)


# %%

# %%
from scipy import optimize
result = optimize.brute(
    m.loss_func, (slice(0, 100, 5
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
