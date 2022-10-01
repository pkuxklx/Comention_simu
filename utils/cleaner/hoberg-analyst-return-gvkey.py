# %%
import pandas as pd
from datetime import datetime 
import numpy as np
import matplotlib.pyplot as plt 
import pickle

# %%

id_type = 'gvkey' # or 'gvkey'

network_path = "data/raw/Hoberg/tnic3_data.txt"
return_path = "data/raw/analyst co-coverage/daily0219.csv"

with open('data/clean/link.p', 'rb') as f:
    link = pickle.load(f)
# %%
# Read csv files

net = pd.read_csv(network_path, sep = '\t')
ret = pd.read_csv('data/raw/analyst co-coverage/daily0219.csv',
                  usecols=['date', 'PERMNO', 'RET']) # Test with 5 rows 
# ret = pd.read_csv(return_path, usecols=['date', 'PERMNO', 'RET'])  
# %%

# Spcify data and stock IDs
date_range = {'start': datetime.strptime('2004-01-01', '%Y-%m-%d'),
              'end': datetime.strptime('2012-12-31', '%Y-%m-%d')}
stock_permno = ret.PERMNO.unique()

# %%
# Select the relevant return data.
ret = ret.loc[ret.PERMNO.isin(stock_permno)]
ret.loc[:, ('date')] = pd.to_datetime(ret.date, format="%Y%m%d")
ret = ret.loc[(ret.date > date_range['start']) &
              (ret.date <= date_range['end'])]

# %%
# We delete the stocks which have NAN in the RET data
select_condition = (ret['RET'] == 'C') | (ret['RET'] == 'B')
ret = ret.drop(ret[select_condition].index)
ret.RET = ret['RET'].astype('float')
ret = ret.pivot_table(values='RET', index='date', columns='PERMNO')
ret_id = ret.columns

print(
    f"# of wanted stock ids {len(stock_permno)}, available stock ids {len(ret_id)}\n {date_range}")


# %% Matching with net 
if id_type == 'gvkey':
    print(f"All the RET_ID has corresponding GVKEY: {all(np.isin(np.unique(ret_id), np.unique(link.PERMNO)))}")

    ret = ret.drop(
        ret_id[~np.isin(np.unique(ret_id), np.unique(link.PERMNO))], axis=1)
    DICT_REPLACE = link.set_index('PERMNO')['GVKEY'].to_dict()
    ret = ret.rename(DICT_REPLACE, axis=1).rename_axis(
        'GVKEY', axis='columns').sort_index(axis=1)

    print(ret.iloc[:5, :5])
    # %% Drop duplicates
    ret = ret.loc[:, ~ret.columns.duplicated()]
    print(
        f"Here we have dropped {(ret.columns.duplicated()).sum()} duplicated stocks")

# %%
