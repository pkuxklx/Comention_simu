# %% 
import pickle 
from wlpy.report import Timer, HyperParams
from wlpy.dataframe import DFManipulation
import pandas as pd
import numpy as np
import torch

def load(hyper_parameter: str):
    """Package the pickle.load function.\n
    There are the following variables loaded: net(network), ret(return), factor, date_range

    Args:
        hyper_parameter (str): the ".json" file to be loaded from.
    """    
    HP = HyperParams(hyper_parameter)
    tt = Timer()
    with open(HP.data_path, 'rb') as file:
        DATA = pickle.load(file)
    print(f'The keys are {DATA.keys()}')
    factor = DATA['factor']
    ret = DATA['ret']
    net = DATA['net']
    return HP, factor, ret, net, tt

# %%

def create_rolling_windows(ret: pd.DataFrame, rolling_parameters: dict, to_dtype = None) -> list:
    """Create a list of rolling windows based on a return df, rolliing window parameters in a dictionary.

    Args:
        ret (pd.DataFrame): The return DF to be separated
        rolling_parameters (dict): Specify the rolling window parameters
        to_dtype ([type], optional): The resulting dtype, can be set to numpy or torch.tensor. Defaults to None.

    Returns:
        [type]: [description]
    """    
    ret_rolling_df = DFManipulation(ret).rolling_window(**rolling_parameters)
    if to_dtype == None:
        return ret_rolling_df 
    elif to_dtype == "numpy":
        ret_rolling_numpy = [i.to_numpy() for i in ret_rolling_df]
        return ret_rolling_numpy
    elif to_dtype == "tensor":
        
        ret_rolling_tensor = [torch.tensor(
            i.to_numpy()) for i in ret_rolling_df]
        return ret_rolling_tensor
    else:
        print("Wrong Dtype, can be df(Default), numpy or tensor.")
        

# %% 
def hokan_and_sort(DF1, Larger_Index):
    """
    This function depends on the order of the index. 
    """
    missing_index = Larger_Index[~np.isin(Larger_Index, DF1.index)]
    v_hokan_df = pd.DataFrame(index = missing_index, columns = DF1.columns).fillna(float(0))
    # Vertical hokan
    DF2 = DF1.append(v_hokan_df).transpose()
    # Horizontal hokan
    h_hokan_df = pd.DataFrame(index = missing_index, columns = DF2.columns).fillna(float(0))
    DF3 = DF2.append(h_hokan_df)
    # vertical and horizontal sort
    DF4 = DF3.sort_index().transpose().sort_index()
    return DF4


def get_G(net: pd.DataFrame, index, year_range=(1996, 2018), aggfunc='mean', id_type="gvkey"):
    if id_type == "gvkey":
        loc_cond = (net.gvkey1.isin(index)) & (net.gvkey2.isin(index))
        t_cond = (net.index.year >= year_range[0]) & (
            net.index.year <= year_range[1])
        G = net.loc[loc_cond & t_cond].pivot_table(
            'score', 'gvkey1', 'gvkey2', fill_value=float(0), dropna=False, aggfunc=aggfunc)
        G[G.columns[~(G.dtypes == 'float64')]] = G[G.columns[~(
            G.dtypes == 'float64')]].astype('float')
        G = hokan_and_sort(G, index)
        return G
    elif id_type == 'permno':
        loc_cond = (net.permno1.isin(index)) & (net.permno2.isin(index))
        t_cond = (net.index.year >= year_range[0]) & (
            net.index.year <= year_range[1])
        G = net.loc[loc_cond & t_cond].pivot_table(
            'number', 'permno1', 'permno2', fill_value=float(0), dropna=False, aggfunc=aggfunc)
        G[G.columns[~(G.dtypes == 'float64')]] = G[G.columns[~(
            G.dtypes == 'float64')]].astype('float')
        G = hokan_and_sort(G, index)
        return G
