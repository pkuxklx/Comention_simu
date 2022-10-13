# %% 
import numpy as np
import pandas as pd 
from wlpy.dataframe import DFManipulation # , RetDF
import matplotlib.pyplot as plt


def report_rslt(port_df):
    dd = {"Standard Deviation": np.std(port_df),
          "Mean Excess Return": np.mean(port_df),
          "Sharpe Ratio": np.mean(port_df)/np.std(port_df)}
    rslt_df = pd.DataFrame(dd)
    port_df.plot()
    print(rslt_df)
    return rslt_df

def compute_weights(sigma_ret):
    import scipy.linalg as LA
    w = LA.inv(sigma_ret).sum(0)/(LA.inv(sigma_ret).sum())
    return w
# %%

