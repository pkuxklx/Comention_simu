# %% 

# This files contains some pre-cleaning that I did. 

import pandas as pd
import pickle

# %%
"""Cleaning the Linking Informtion"""
link = pd.read_csv("data/raw/linking_information.csv")
link = link.loc[:,('gvkey', 'LPERMNO')]
link = link.rename(columns = {'gvkey': "GVKEY", 'LPERMNO': 'PERMNO'})

with open("data/clean/link.p", 'wb') as f:
    pickle.dump(link, f)

# %% Check if permno maps uniquely to gvkey, the answer is no. 

# for j in link.PERMNO:
    # cond = (link.PERMNO == j)
    # aa = link.loc[cond, 'GVKEY']
    # if len(aa) >1 :
        # print(j)
# %%
'''Handling of the Factors'''
factor_path = "data/raw/factor/factors_2021-01-13.csv"

factor = pd.read_csv(factor_path, index_col='date')
factor.index = pd.to_datetime(factor.index, format='%Y%m%d')

with open("data/clean/factor.p", 'wb') as f:
    pickle.dump(factor, f)
# %%