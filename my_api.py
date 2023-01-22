# %%
import numpy as np
import matplotlib.pyplot as plt
import random, os
from infoband.band_info import InfoCorrBand

# %%
def cov2cor(S: np.ndarray):
    # Covariance to Correlation
    D = np.diag(np.sqrt(np.diag(S)))
    D_inv = np.linalg.inv(D)
    return D_inv @ S @ D_inv

# %%
def gen_S_AR1(rho = 0.8, N = 500) -> np.ndarray:
    # self covariance matrix of AR(1) process
    S_block = np.zeros(shape=[N, N])
    for j in range(0, N):
        S_block = S_block + np.diag(np.ones(N - j) * (rho ** j), -j) + np.diag(np.ones(N - j) * (rho ** j), j)
    S = S_block - np.eye(N)
    return S

# %%
def gen_eta_sequence(N, eta = 0.5, draw_type = 'random', is_random = False, 
                     rand_seed = 100, near_factor = 2) -> np.ndarray:
    """
    Generate a sequence b, which is a permutation of {1, ..., N}. 
    b satisfies the property: for any 0 < k < N+1, b[0]~b[k-1] include {1, ..., ceil(eta*k)}.  
    
    draw_type : {'random', 'near'}
        Algorithms about how to draw ( {b[0], ..., b[k-1]} - {1, ..., ceil(eta*k)} ). Here '-' is a subtraction between two sets.
    is_random : bool
        If False, we use random_seed as random seed, for repeat running results.
    random_seed : int
    near_factor : float
        Needed only when draw_type = 'near'.
    """
    if is_random:
        rng = random
    else:
        rng = np.random.RandomState(rand_seed)
        
    b = [1] # Default to keep the diagonal element in covariance estimation.
    b_complement = [i for i in range(2, N + 1)] # b's complement set
    
    for k in range(2, N + 1):
        # consider k-th element
        th = int(np.ceil(eta * k))
        # S^L_k include S^d_{th}
        cnt = sum([1 if num <= th else 0 for num in b])
        if cnt < th:
            for next_id in range(1, th + 1):
                if next_id not in b:
                    b.append(next_id)
                    b_complement.remove(next_id)
                    break
        else:
            # len(b_complement) == N + 1 - k
            if draw_type == 'random':
                j = rng.randint(0, N - k) if N - k > 0 else 0
            elif draw_type == 'near':
                upper = min(int(near_factor * k), N - k)
                j = rng.randint(0, upper) if upper > 0 else 0
            else:
                raise Exception('draw_type, value error')
            next_id = b_complement[j] 
            b.append(next_id)
            b_complement.remove(next_id)
    return np.array(b)

# %%
def gen_L(S, eta, verbose = False, draw_type = 'random', is_random = False, 
          rand_seed = 100, near_factor = 2):
    N = S.shape[0]
    new_rowSort = np.zeros((N, N))
    
    R = cov2cor(S)
    L = abs(R)
    rowSort = InfoCorrBand(X = np.eye(N), L = L).rowSort # You can ignore the 'X = np.eye(N)' parameter. I create this temporary object solely to get 'rowSort' matrix.
    
    for i in range(N):
        row = rowSort[i]
        argst = row.argsort()
        b = gen_eta_sequence(N, eta, draw_type, is_random, rand_seed, near_factor)
        for j in range(N):
            new_rowSort[i][argst[j]] = b[j]
    
    L_eta = 1 / new_rowSort
    res = (L_eta, new_rowSort, rowSort)
    return res if verbose else L_eta

# %%
def get_title_1(ord, cov_cor, eta, N, T, rho, draw_type = 'random', near_factor = None):
    title = "{ord}, {cov_cor} error, {draw_type}, eta=({eta}, {near_factor}), (N, T)=({N}, {T}), rho={rho}".format(
        ord = ord, cov_cor = cov_cor, draw_type = draw_type, eta = eta, N = N, T = T, rho = rho, near_factor = near_factor)
    return title

# %%
def save_data_fig(x, ord, cov_cor, draw_type, eta, N, T, rho, near_factor = None, 
                  is_save = False):
    x = list(x)
    
    title = get_title_1(ord, cov_cor, eta, N, T, rho, draw_type, near_factor)
    data_path = 'data/' + title + '.txt'
    fig_path = 'data/' + title + '.png'
    print(data_path)
    
    old_x = np.loadtxt(fname = data_path, ndmin = 1).tolist() if os.path.exists(data_path) else []
    full_x = x + old_x
    size = len(full_x)
    
    plt.figure(figsize = (10, 1))
    plt.hist(x = full_x, bins = 100, color = '#0504aa', alpha = 0.7, rwidth = 0.85)
    plt.title(title + ", size={}".format(size))
    plt.xlabel("error rate")
    plt.ylabel("frequency")
    
    if is_save:
        plt.savefig(fig_path, bbox_inches = 'tight')
        np.savetxt(fname = data_path, X = full_x)