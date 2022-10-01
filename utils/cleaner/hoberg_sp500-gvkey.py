# %%
#  Created on 19 November 2020
#  Author: Weiguang Liu
"""
This file will prepare the Hoberg Network based on a Return DataFrame
The end result will be a list of network for each year and confroming T*N dataframe containing returns. 
"""

import pandas as pd
import numpy as np
import pickle
# %%

save_path = "data/clean/hoberg-sp500-gvkey.p"

network_path = "data/raw/Hoberg/tnic3_data.txt"
return_path = "data/raw/return/sp500-return-2021-01-21.csv"
factor_path = "data/raw/factor/factors_2021-01-13.csv"
link_path = "data/raw/linking_information.csv"

# %% Import the csv file
net = pd.read_csv(network_path, sep='\t')
ret = pd.read_csv(return_path)
ret.date = pd.to_datetime(ret.date, format='%Y%m%d')
net['year'] = pd.to_datetime(net.year, format='%Y')
net = net.set_index('year')
print(net.iloc[:5, :5], '\n', ret.iloc[:5, :5])
# %% Alignment in time
year_start = min(ret.date.dt.year)
year_end = max(net.index.year)
net = net[net.index.year >= year_start]
print(net.iloc[:5, :5], '\n')
print(net.iloc[-5:, :5], '\n')
# NETWORK[pd.isna(NETWORK['score'])]
# %%
# We delete the stocks which have NAN in the RET data
select_condition = (ret['RET'] == 'C') | (ret['RET'] == 'B')
ret = ret.drop(ret[select_condition].index)
# %%
ret.RET = ret['RET'].astype('float')
ret = ret.pivot_table(values='RET', index='date', columns='PERMNO')
ret_id = ret.columns
retp = ret
with open("data/clean/sp500-ret-permno.p", 'wb') as file:
    pickle.dump(retp, file)

# %%
# ================================
# Construct a dictionary that maps PERMNO to GVKEYS
# ================================
LIST = np.unique(net['gvkey1'])
# This is used to download mapping file from CRSP.
# LIST.tofile('List.txt', sep='\n')
# %%
# We change the RET code from PERMNO to GVKEY

link = pd.read_csv(link_path)
link = link.iloc[:, [0, 5]]
link = link.rename(columns={link.columns[0]: 'GVKEY'})
print(link.iloc[:5, :5])
# %%
# Check that all the RET_ID has corresponding GVKEY

all(np.isin(np.unique(ret_id), np.unique(link.LPERMNO)))
ret = ret.drop(
    ret_id[~np.isin(np.unique(ret_id), np.unique(link.LPERMNO))], axis=1)
# %%
DICT_REPLACE = link.set_index('LPERMNO')['GVKEY'].to_dict()
ret = ret.rename(DICT_REPLACE, axis=1).rename_axis(
    'GVKEY', axis='columns').sort_index(axis=1)
print(ret.iloc[:5, :5])
# %% Drop duplicates
ret = ret.loc[:, ~ret.columns.duplicated()]
print(f"Here we have dropped {(ret.columns.duplicated()).sum()} duplicated stocks")
# %%
# We select the stocks in RET and construct the network matrices among them
ret_id = ret.columns
net = net[net.gvkey1.isin(ret_id)]

condition1 = net.reset_index().set_index(['gvkey1']).index.isin(ret_id)
condition2 = net.reset_index().set_index(['gvkey2']).index.isin(ret_id)
net = net.loc[condition1 & condition2]


# %%
'''Handling of the Factors'''
factor = pd.read_csv(factor_path, index_col='date')
factor.index = pd.to_datetime(factor.index, format='%Y%m%d')
condition = (factor.index.year >= year_start) & (factor.index.year <= year_end)
factor = factor.loc[condition]

# %%

DATA = {'ret': ret, 'net': net, 'factor': factor,
        'year_start': year_start, 'year_end': year_end}
with open(save_path, 'wb') as file:
    pickle.dump(DATA, file)


# %%

