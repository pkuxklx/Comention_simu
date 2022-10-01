import pandas as pd
import numpy as np


def import_network(file_path, set_index=False, reporter = None):
    NET= pd.read_csv(file_path, sep = '\t')
    NET['year'] = pd.to_datetime(NET.year, format = '%Y').dt.normalize()
    if set_index:
        NET = NET.set_index('year')
    return NET

def import_return(file_path, option = 'gvkey-all', reporter = None):
    """
    options: 'gvkey-all' will import return data from [return-2021-01-12.csv] \n
    """
    if reporter == None:
        jot = print
    else:
        jot = reporter.jot
    
    if option == 'gvkey-all':        
        PRICE = pd.read_csv(file_path,  usecols=['datadate', 'GVKEY', 'prccd']).rename(columns={'GVKEY': 'gvkey', 'datadate': 'date'})
        jot(f'We import date, gvkeys and closing prices from the dataframe.')
        PRICE.date = pd.to_datetime(PRICE.date, format = '%Y%m%d').dt.normalize()
        PRICE = PRICE.sort_values(by = ['gvkey','date'])
        if PRICE.prccd.dtype != 'float64':
            print('price data are not of dtype float, need correction')
        PRICE = PRICE.pivot_table(index = 'date', columns='gvkey', values='prccd', aggfunc = np.mean)
        jot(f'We have combined the multiple security returns by the average.')
        RET = PRICE.pct_change(1).drop(PRICE.index[0])
    else:
        RET = pd.DataFrame([])
    return RET
