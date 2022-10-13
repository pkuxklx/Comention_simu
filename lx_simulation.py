# %%
import numpy as np
from infoband.band_info import InfoCorrBand
# %%
rng = np.random.RandomState(100)
N = 100
T = 80
S = sum( [make_sparse_spd_matrix(N, random_state = i) for i in range(5)] )
print(S[:5, :5])
X = rng.multivariate_normal(mean =np.zeros(N), cov = S, size = T)
# %%
c = InfoCorrBand(X)
# c.sample_cov()[:3, :3]
# c.sample_corr()[:3, :3]
# %%
def gen_S(rho = 0.8,N = 500):
    S_block = np.zeros(shape=[N, N])
    for j in range(0, N):
        S_block = S_block + np.diag(np.ones(N-j)*(rho**j), -j) + \
        np.diag(np.ones(N-j)*(rho**j), j)
    S = S_block - np.eye(N)
    return S
# %%
L = abs(S)
c.feed_info(L)
# c.plot_k_pd()
# %%
c.find_biggest_k_for_pd()
# %%
# c.plot_k_pd(range(N-50, N+1))
# %%
k = c.k_by_cv(verbose = False)
k
# %%
R_est = c.fit_info_corr_band(k)
S_est = c.fit_info_cov_band(k)
# %%