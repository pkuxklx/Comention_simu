# %%
import numpy as np
import copy
from wlpy.gist import heatmap

# %%
def cov2cor(S: np.ndarray):
    # Covariance to Correlation
    D = np.diag(np.sqrt(np.diag(S)))
    D_inv = np.linalg.inv(D)
    return D_inv @ S @ D_inv

# %%
def gen_S_AR1(rho: float, N: int, cut: int = None) -> np.ndarray:
    # self covariance matrix of AR(1) process
    S_block = np.zeros(shape = [N, N])
    for j in range(N):
        S_block = S_block + np.diag(np.ones(N - j) * (rho ** j), -j) + np.diag(np.ones(N - j) * (rho ** j), j)
    S = S_block - np.eye(N)
    if cut:
        i_indices, j_indices = np.indices((N, N))
        S[np.abs(i_indices - j_indices) > cut] = 0
    return S
# %%
def gen_S_Cai2011Adaptive_Model1(N: int, t: int = 10, a1: int = 1, a2: int = 4) -> np.ndarray:
    """
    Args:
        t: Bandwidth of the block matrix A1.
        ai: amplitude

    Returns:
        numpy.ndarray: 
    """
    Nh = N // 2
    A1 = np.zeros(shape = (Nh, Nh))
    for j in range(Nh):
        sigma = max(0, 1 - j / t) * a1
        A1 = A1 + np.diag(np.ones(Nh - j) * sigma, -j) + np.diag(np.ones(Nh - j) * sigma, j)
    A1 = A1 - np.eye(Nh)
    A2 = np.eye(Nh) * a2
    S = np.zeros(shape = (N, N))
    S[0:Nh, 0:Nh] = A1
    S[Nh:N, Nh:N] = A2
    return S

if __name__ == '__main__':
    S = gen_S_Cai2011Adaptive_Model1(100)
    heatmap(S)
# %%
def gen_S_Cai2011Adaptive_Model2(N: int, intB: tuple = (0.3, 0.8), probB: int = 0.2, a2: int = 4, seed: int = None) -> np.ndarray:
    """
    Args:
        intB: The range of uniform distribution.
        probB: The probability in Bernoulli distribution.

    Returns:
        numpy.ndarray: 
    """
    Nh = N // 2
    rng = np.random.RandomState(seed) if seed else np.random
    A2 = np.eye(Nh) * a2
    B = rng.uniform(low = intB[0], high = intB[1], size = (Nh, Nh)) * rng.binomial(n = 1, p = probB, size = (Nh, Nh))
    eigvals, _ = np.linalg.eig(B)
    eps = max(0, - min(eigvals)) + 0.01
    A1 = B + eps * np.eye(Nh)
    S = np.zeros(shape = (N, N))
    S[0:Nh, 0:Nh] = A1
    S[Nh:N, Nh:N] = A2
    return S

if __name__ == '__main__':
    S = gen_S_Cai2011Adaptive_Model2(100, intB = (0, 6), a2 = 2)
    heatmap(S)

# %%
def gen_S_Bernoulli():
    # TODO
    pass