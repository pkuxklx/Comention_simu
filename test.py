# %%
import numpy as np
from utils.my_api import get_title_1
# %%
path = './data_plat/'
name = get_title_1('fro', 'cov', 'random', 0.5, 500, 500, 0.8)
x = np.loadtxt(path + name + '.txt') 
# %%
x.argmin()
# %%
x.argmax()
# %%
