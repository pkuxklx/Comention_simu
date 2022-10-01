# %%
# Created on 12 January 2021
# Author: Weiguang Liu
"""
This file will prepare the Hoberg Network based on a Return DataFrame
The end result will be a list of network for each year and confroming T*N dataframe containing returns. 
"""
import numpy as np
import pandas as pd
from importlib import reload
import pickle
from utils import raw_dti, raw_dto 
from wl_report import DataReport
# %%

output_name = 'data/clean/hoberg-gvkey-all.p'

raw_folder = 'data/raw'

# Path to the data files
net_path = f'{raw_folder}/Hoberg/tnic3_data.txt'
factor_path = f'{raw_folder}/factor/factors_2021-01-13.csv'
ret_path = f'{raw_folder}/return/return-2021-01-12.csv'


# Initialize report
drpt = DataReport(output_name + '.md')
drpt.copy(net_path + '.md')
drpt.copy(ret_path + '.md')
drpt.copy(factor_path + '.md')
jot = drpt.jot

# %%
net = raw_dti.import_network(net_path, set_index=True, reporter = drpt)
year_start = min(net.index.year)
year_end = max(net.index.year)
drpt.df_head(net)


ret = raw_dti.import_return(ret_path, option = 'gvkey-all', reporter =drpt)
ret = ret.loc[(ret.index.year >= year_start) & (ret.index.year <= year_end)]
drpt.df_head(ret)
# %%
'''We want to construct network based on the return data'''
ret_id = ret.columns
condition1 = net.reset_index().set_index(['gvkey1']).index.isin(ret_id)
condition2 = net.reset_index().set_index(['gvkey2']).index.isin(ret_id)
net = net.loc[condition1 & condition2]
jot(f'We keep the network members that are in the return dataframe.')


# %%
'''Handling of the Factors'''
factor = pd.read_csv(factor_path, index_col='date')
factor.index = pd.to_datetime(factor.index, format='%Y%m%d')
condition = (factor.index.year >= year_start) & (factor.index.year <= year_end)
factor = factor.loc[condition]

# %%
DATA = {'ret': ret, 'net': net, 'factor': factor,
        'year_start': year_start, 'year_end': year_end}
raw_dto.save_result(DATA, output_name)
