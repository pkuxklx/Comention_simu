# %%
import pandas as pd
import numpy as np
import seaborn as sns
import pickle

# %%


output_name = "data/clean/ibes_0419.p"

retpath = "data/raw/analyst co-coverage/return_panel_0419.csv"
network_path = "data/raw/analyst co-coverage/W_IBES0419.csv"
factor_path = "data/raw/analyst co-coverage/F-F_Research_Data_5_Factors_2x3_daily.CSV"
ret = pd.read_csv(retpath, index_col = 0)
net = pd.read_csv(network_path, index_col = 0)
factor = pd.read_csv(factor_path, index_col = 0)
factor.index = pd.to_datetime(factor.index, format="%Y%m%d")

# %%
net = net.rename(lambda a: int(a.replace('V', '')), axis = 'columns')
ret = ret.rename(lambda a: int(a.replace('V', '')), axis = 'columns')

# %%
factor = factor.loc[(factor.index.year <= 2019) &(factor.index.year >=2004)]
dd = {i: j for i, j in zip(factor.columns, ['mktrf', 'smb', 'hml', 'rmw', 'cma', 'rf'])}
factor = factor.rename(dd, axis = 'columns')

# %%

ret.reset_index().drop('index', axis= 1)
ret.index = factor.index.copy()

# %%
aa = net.to_numpy().flatten()

year_start = 2004
year_end = 2019

DATA = {'ret': ret, 'net': net, 'factor': factor,
        'year_start': year_start, 'year_end': year_end}
with open(output_name, 'wb') as file:
    pickle.dump(DATA, file)

# %%
