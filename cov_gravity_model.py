# %%
import numpy as np
from wlpy.gist import heatmap
rng = np.random.default_rng()
# %%
def normalize_covariance(S):
    D = np.diag(np.diag(S)**-0.5)
    return D @ S @ D

def mass(N, k = 5, max_mass = 0.5):
    M = rng.noncentral_chisquare(k, False, size = N)
    # M = rng.exponential(1, size = N)
    # M = rng.uniform(0, 1, size = N)
    M = np.sort(M)
    M = M / np.max(M) * max_mass
    return M

def gen_cov(N, mass, poly_param = [0.8, 0.5], normalize = False):
    """Generate a covariance matrix with gravity model, with diagonal elements equal to 1.
    $y = (I + b_1 G + b_2 G^2) e$ for a network matrix $G$

    Args:
        N: the dimension of the covariance matrix
        mass: the vector containing the mass of $N$ nodes

    Returns:
        covariance : a N by N covariance matrix
    """
    prob_of_link = np.array([[x * y for x in mass] for y in mass])

    network = rng.binomial(1, prob_of_link)
    network = network - np.diag(np.diag(network))

    pre_cov = np.eye(N) + poly_param[0] * network + \
        poly_param[1] * (network @ network).clip(0, 1)

    if normalize:
        cov = normalize_covariance(pre_cov @ pre_cov.T)
    else:
        cov = pre_cov @ pre_cov.T
    return cov

def gen_aux_set(cov, prob, qrob, observe_level):
    P = np.random.binomial(1, prob, size = cov.shape)
    Q = np.random.binomial(1, qrob, size = cov.shape)

    GP = np.where((cov > observe_level) & (P == 1), 1, 0)
    GQ = np.where((cov <= observe_level) & (Q == 1), 1, 0)

    aux_set = GP + GQ
    return aux_set

# %%
# Nonconcentral chisquare
import matplotlib.pyplot as plt
M = mass(2000000, k = 5, max_mass = 0.5)
plt.hist(M, bins = 200);
# %%

# %%
# Test case
nsim = 2
sample_size = 500 # N by N matrix

prob = 0.8
qrob = 0.2
observe_level = 0.5

for _iter in range(nsim):
    M = mass(sample_size, df=5, max_mass=0.5)
    cov = gen_cov(sample_size, M, poly_param=[0.8, 0.5])
    heatmap(cov)
    heatmap(cov[0: 50, 0: 50])
    aux_set = gen_aux_set(cov, prob=prob, qrob=qrob,
                          observe_level=observe_level)
    
# %%
