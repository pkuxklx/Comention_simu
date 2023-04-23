# %%
from my_api import *
import numpy as np
# %%
for N in [100, 300, 500]:
    print(N)
    S = gen_S_Cai2011Adaptive_Model2_my(N, seed = 0)
    for i in range(1, 21):
        Si = gen_S_Cai2011Adaptive_Model2_my(N, seed = i)
        from numpy.linalg import norm
        for ord in ['fro', 2]:
            print(f'{norm(Si, ord) / norm(S, ord) :.2f}, {ord}')
# %%
