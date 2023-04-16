import numpy as np
import scipy.linalg as LA 
# from utils.covest import NetBanding
import pandas as pd

def gen_S(rho = 0.99,num_blocks = 1, each_block_size= 200):
    N = each_block_size
    Z = np.zeros([N,N])
    S_block = np.zeros(shape=[N, N])
    for j in range(0, N):
        S_block = S_block + np.diag(np.ones(N-j)*(rho**j), -j) + \
        np.diag(np.ones(N-j)*(rho**j), j)
    S = S_block - np.eye(N)
    return S

def func_G(R, prob, qrob, observe_level, seed = None):
    rng = np.random.default_rng(seed)
    P = rng.binomial(1, prob, size = R.shape)
    Q = rng.binomial(1, qrob, size = R.shape)
    P = np.triu(P) + np.triu(P, 1).T
    Q = np.triu(Q) + np.triu(Q, 1).T

    R = np.abs(R)
    GP = np.where((R > observe_level) & (P == 1), 1, 0)
    GQ = np.where((R <= observe_level) & (Q == 1), 1, 0)

    G = GP + GQ
    return G

def generate_sample(S, T = 200):
    rng= np.random.default_rng()
    N = S.shape[0]
    X1 = rng.multivariate_normal(mean=np.zeros(N), cov=S, size=T)
    return X1


def norm_rslt(S, m, S_new_lst=[], norm_type='fro'):
    norm_reslt = [LA.norm(S, ord=norm_type),
                  LA.norm(m.S_sample - S, ord=norm_type),
                  LA.norm(m.lin_shrink() - S, ord=norm_type),
                  LA.norm(m.nonlin_shrink() - S, ord=norm_type)] + [LA.norm(j - S, ord=norm_type) for j in S_new_lst]
    return norm_reslt


def generate(N, T, rho, prob, qrob, observe_level):
    R = gen_S(rho, each_block_size=N)
    S = R * 3
    G = func_G(R, prob=prob, qrob=qrob, observe_level=observe_level)
    X = generate_sample(S, T)
    return R, S, G, X


def estimate(R, S, G, X, param, uni_param = None, norm_type = 'fro'):
    if uni_param == None:
        uni_param = param
   
    T, N = X.shape 
    return rslt

def rslt_df(lst):

    df = pd.DataFrame(lst)
    param_name = ['$rho$',
                'p',
                'q',
                'Observation Level',
                'Threshold Level',
                'S']
    estimator_name = ['Sample Cov',
                    'Linear Shrinkage',
                    'Nonlinear Shrinkage',
                    'Universal Threshold',
                    'Network Guided']
    df.columns = param_name + estimator_name
    return df, param_name, estimator_name
