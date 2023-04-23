# %%
import numpy as np

# %%
with open('LSRthreshold_log.txt') as fp:
    lines = fp.readlines()

for s in lines[:3]:
    print(s, end = '')
lines = lines[3:]

# %%
th = [float(s.split(': ')[1][:4]) for s in lines]

err = [float(s.split(': ')[2][:4]) for s in lines]

abserr = [float(s.split(': ')[3][:4]) for s in lines]
# %%
X = list(zip(th, err))
np.corrcoef(x = X, rowvar = False)
# %%
X = list(zip(th, abserr))
np.corrcoef(x = X, rowvar = False)
# %%
abserr = np.array(abserr)
print(abserr.mean(), abserr.std())
# %%
