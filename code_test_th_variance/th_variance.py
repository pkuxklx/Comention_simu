# %%
import numpy as np
import matplotlib.pyplot as plt
# %%
x = np.loadtxt(fname = 'Cai2011Model2_LSR_th.txt')
y = np.loadtxt(fname = 'Cai2011Model2_hardthres_th.txt')
# %%
alpha = 0.3
a, b, step = 0.8, 2, 0.05
bins = np.linspace(a, b, 1 + int((b - a) // step))
plt.hist(x, color = 'red', alpha = alpha, bins = bins, label = 'Augment before Threshold')
plt.hist(y, color = 'blue', alpha = alpha, bins = bins, label = 'Hard Threshold')
plt.xlabel('threshold chosen by cv')
plt.legend(title = 'N=500, T=300, Frobenius norm, True Cov=Cai2011Adaptive_Model2_my')
plt.show()
# %%
