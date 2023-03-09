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
def gen_eta_sequence(N, eta = 0.5, draw_type = 'random', seed = None, near_factor = 2) -> np.ndarray:
    """Generate a sequence b, which is a permutation of {1, ..., N}. 
    b satisfies the property: for any 0 < k < N+1, b[0]~b[k-1] include {1, ..., ceil(eta*k)}.  
    
    Args:
        draw_type : {'random', 'near'}. Algorithms about how to draw ({b[0], ..., b[k-1]} - {1, ..., ceil(eta*k)}).
        seed: int or None
        near_factor: float
        Needed only when draw_type = 'near'.
        
    Returns:
    """
        
    rng = np.random.RandomState(seed) if seed else np.random
        
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
def gen_L(S, eta, verbose = False, draw_type = 'random', near_factor = 2, seed = None):
    '''
    Auxiliary set for the augmented banding method
    
    Args:
        seed: seed for numpy.random.RandomState
    
    Returns:
        L matrix, which records the order of magnitudes for each row of the covariance S
    '''
    N = S.shape[0]
    new_rowSort = np.zeros((N, N))
    
    R = cov2cor(S)
    L = abs(R)
    rowSort = InfoCorrBand(X = np.eye(N), L = L).rowSort # You can ignore the 'X = np.eye(N)' parameter. I create this temporary object solely to get 'rowSort' matrix.
    
    for i in range(N):
        row = rowSort[i]
        argst = row.argsort()
        b = gen_eta_sequence(N, eta, draw_type, seed, near_factor)
        for j in range(N):
            new_rowSort[i][argst[j]] = b[j]
    
    L_eta = 1 / new_rowSort
    res = (L_eta, new_rowSort, rowSort)
    return res if verbose else L_eta

# %%
def get_file_name(ord, cov_cor, N, T, covariance_describe, simu_describe):
    """
    Args:
        covariance_describe: 
            ['AR1', rho] or ['Grav', max_mass]
        simu_describe: 
            ['lx_band', eta] or ...
    """
    title = "{},{},{},{},{covariance_describe},{simu_describe}".format(
        ord, cov_cor, N, T, 
        covariance_describe = covariance_describe, 
        simu_describe = simu_describe)
    return title

# %%
get_file_name(2, 'S', 100, 300, ['AR1', 0.8], [0.5])

# %%
def save_data_fig(x, 
                  ord, cov_cor, N, T, 
                  cov_dscrb, simu_dscrb, 
                  folder = 'data', is_save = False):
    """
    Args:
        folder: string or None, the folder to save files
    """
    x = list(x)
    
    title = get_file_name(ord, cov_cor, N, T, cov_dscrb, simu_dscrb) 
    data_path = os.path.join(folder, title + '.txt')
    fig_path = os.path.join(folder, title + '.png')
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

# %%

# Weiguang Liu's api

# %%
def gen_mass(N, dof = 5, max_mass = 0.5, seed = None):
    rng = np.random.RandomState(seed) if seed else np.random
    mass = rng.noncentral_chisquare(dof, False, size = N)
    # mass = rng.exponential(1, size = N)
    # mass = rng.uniform(0, 1, size = N)
    
    # mass = np.sort(mass)
    mass = mass / np.max(mass) * max_mass
    return mass

def gen_S_gravity(N, dof = 5, max_mass = 0.5, 
                  poly_param = [0.8, 0.5], 
                  heteroskedasticity = True, 
                  seed = None, verbose = False):
    """Generate a covariance matrix with gravity model, with diagonal elements equal to 1.
    $y = (I + b_1 G + b_2 G^2) e$ for a network matrix $G$

    Args:
        N: the dimension of the covariance matrix
        mass: the vector containing the mass of $N$ nodes

    Returns:
        a N by N covariance matrix
    """
    rng = np.random.RandomState(seed) if seed else np.random
    
    mass = gen_mass(N, dof, max_mass, seed)
    prob_of_link = np.array([[x * y for x in mass] for y in mass])
    network = rng.binomial(1, prob_of_link)
    network = network - np.diag(np.diag(network))
    pre = np.eye(N) + poly_param[0] * network + \
        poly_param[1] * (network @ network).clip(0, 1)
    R = cov2cor(pre @ pre.T)
    
    if heteroskedasticity:
        sigma = rng.normal(loc = np.ones(N), scale = 0.3).clip(0.2, None)
        D = np.diag(sigma)
    else:
        D = np.ones(N) 
    S = D @ R @ D
        
    if verbose:
        heatmap(prob_of_link, 'prob_of_link');
        heatmap(network, 'network');
        heatmap(pre, 'pre');
        heatmap(R, 'cor');
        heatmap(D, 'D');
        heatmap(S, 'cov');
        
    return S
# %%
def gen_aux_set(cov, prob, qrob, observe_level):
    P = np.random.binomial(1, prob, size = cov.shape)
    Q = np.random.binomial(1, qrob, size = cov.shape)

    GP = np.where((cov > observe_level) & (P == 1), 1, 0)
    GQ = np.where((cov <= observe_level) & (Q == 1), 1, 0)

    aux_set = GP + GQ
    return aux_set

# %%

# %%
class MyParamsIter:
    def __init__(self, *args):
        self.ls = args
    
    def __lt__(self, x): # <
        assert type(x) == MyParamsIter
        assert len(self.ls) == len(x.ls)
        fro_to_0 = lambda x: 0 if x == 'fro' else x
        for v1, v2 in zip(self.ls, x.ls):
            v1 = fro_to_0(v1)
            v2 = fro_to_0(v2)
            assert type(v1) != str and type(v2) != str
            if v1 == v2:
                continue
            return v1 < v2
        return False     

    def __le__(self, x):
        return self < x or self.ls == x.ls