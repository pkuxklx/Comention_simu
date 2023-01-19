# %%
import pandas as pd
from utils.covest import NetBanding
import numpy as np
from utils import simulation
from wlpy.gist import heatmap, generalized_threshold
from wlpy.covariance import Covariance
from scipy import linalg as LA
import matplotlib.pyplot as plt
# %%
N = 200
T = 100
rho = 0.9

# std = np.diag(np.random.uniform(1,5, N)**0.5)
R = simulation.gen_S(rho, each_block_size = N)
# S = std @ R @std
S = R*3

heatmap(S)

X = simulation.generate_sample(S, T=T)
G = simulation.func_G(R, prob=1, qrob=0.2, observe_level=0.5)
heatmap(G)


# %%

m = NetBanding(X, G, N, T)
print(simulation.norm_rslt(S, m))

# %%
# pppp= m.params_by_cv('pd')
# print(pppp)
# print(LA.norm(m.fit(pppp) - S, ord=1))

# %% 
# ==========
# [1]: Chaning rho and threshold level
# ==========
lst = []

num_endpoints = 11
norm_type = 'fro'
norm_dict = {2: 'Operator Norm', 1: 'Matrix 1-Norm', 'fro': 'Frobenius Norm'}

changing_parameter = "Threshold Level"

for j in [0.7,0.8,0.9,0.95, 0.99]:
    rho, prob, qrob, observe_level = j, 0.9, 0, 0.25
    R, S, G, X = simulation.generate(N, T, rho, prob, qrob, observe_level)
    for i in np.linspace(0, 1, num_endpoints):
        param = i
        uni_param = param
        
        m = NetBanding(X, G, N, T, use_correlation=True)
        S_new_lst = [m.fit(uni_param, option='universal'), m.fit(param)]
        rslt = [rho, prob, qrob, observe_level, param] + \
        simulation.norm_rslt(S, m, S_new_lst, norm_type=norm_type)

        lst += [rslt]

df, param_name, estimator_name = simulation.rslt_df(lst)

df1 = df.set_index(['$rho$', 'Threshold Level']).iloc[:,4:]
caption = f"The estimation error of various estimators in terms of the {norm_dict[norm_type]}"
with open(f"asset/table2-{norm_type}.tex", 'w') as f :
    f.write(df1.to_latex(float_format="%.2f",
            column_format="lp{2cm}|p{2cm}p{2cm}p{2cm}p{2cm}p{2cm}", label=f't:2-{norm_type}', longtable = True, caption = caption))


print(df1)
# %%
# ==========
# [2] When we vary the observation level 
# ==========
lst = []

num_endpoints = 51
norm_type = 2

rho, prob, qrob = 0.95, 1, 0
R = simulation.gen_S(rho = rho, each_block_size=N)
S = R * 3
X = simulation.generate_sample(S, T)

for j in np.linspace(0,1, num_endpoints):
    observe_level = j
    G = simulation.func_G(R, prob=prob, qrob=qrob, observe_level=observe_level)

    param = 0.3
    uni_param = param

    m = NetBanding(X, G, N, T, use_correlation=True)
    S_new_lst = [m.fit(uni_param, option='universal'), m.fit(param)]
    rslt = [rho, prob, qrob, observe_level, param] + \
        simulation.norm_rslt(S, m, S_new_lst, norm_type=norm_type)

    lst += [rslt]

df, param_name, estimator_name = simulation.rslt_df(lst)

fig, ax = plt.subplots()
[ax.plot(np.linspace(0,1, num_endpoints), df[j]) for j in estimator_name]
ax.legend(estimator_name, loc='best', fontsize = 8)
ax.set_ylabel(f"{norm_dict[norm_type]} Estimation Error", fontsize=12)
ax.set_xticks(np.linspace(0,1,11))
ax.set_xlabel('Observation Level', fontsize=12)
fig.savefig(f"asset/{'Observation Level'.lower().replace(' ', '-')}-{norm_type}.eps", format="eps")
# %%
# ==========
# [3] Varing p and q
# ==========
num_endpoints = 11
norm_type = 'fro'
rho = 0.95
observe_level = 0.2

R = simulation.gen_S(rho=rho, each_block_size=N)
S = R * 3
X = simulation.generate_sample(S, T)
lst = []
for i in np.linspace(0,1, num_endpoints):
    for j in np.linspace(0,1, num_endpoints):
        prob = i
        qrob = j
        G = simulation.func_G(R, prob= i, qrob = j, observe_level = observe_level)
        
        param = 0.3
        uni_param = param

        m = NetBanding(X, G, N, T, use_correlation=True)
        S_new_lst = [m.fit(uni_param, option='universal'), m.fit(param)]
        rslt = [rho, prob, qrob, observe_level, param] + \
            simulation.norm_rslt(S, m, S_new_lst, norm_type=norm_type)

        lst += [rslt]
df, param_name, estimator_name = simulation.rslt_df(lst)
df1= df.pivot_table(values = 'Network Guided', index = 'p', columns= 'q')
caption = f"The estimation error in terms of {norm_dict[norm_type]} of the Network Guided Estimator with varying probabilities $p$, $q$ that determine how $G$ is generated."

df1
with open(f"asset/table3-{norm_type}.tex", 'w') as f:
    f.write(df1.to_latex(float_format="%.2f", label=f't:3-{norm_type}', longtable = True, caption=caption))
# %%
