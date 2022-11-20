# %%
import numpy as np
import matplotlib.pyplot as plt
# %%
full_x = np.loadtxt(fname = 'testscore2') # 下标从1开始
ans = full_x.argmin() + 1
# %%
def __loss_func(k):
    return full_x[k-1]

__loss_func(12)
N = len(full_x)
# %%  


# %%
def moving_average(arr, win = 5, alpha = 0.8):
    '''
    win: 
        length of the moving window
    alpha: 
        decay coefficient
    '''
    c = (win + 1) / 2 - 1 # weight[c], the central index
    weight = np.array([alpha ** abs(c - i) for i in range(win)])
    weight = weight / weight.sum()
    l = len(arr)
    new_arr = [0] * l
    for i in range(l):
        for k, w in enumerate(weight):
            id = int(min(max(i + k - c, 0), l - 1))
            new_arr[i] += w * arr[id]
    return new_arr
# %%
k_lower = 1
k_upper = N
delta = N // 4

while 1:
    k = k_lower
    k_list, k_score = [], []
    while k <= k_upper:
        k_score.append(__loss_func(k))
        k_list.append(k)
        k = k + 1 if k == k_upper else min(k + delta, k_upper)
    id = np.array(k_score).argmin()
    min_k = k_list[id]
    if delta == 1:
        MA_k_score = moving_average(k_score)
        MA_id = np.array(MA_k_score).argmin()
        min_k = k_list[MA_id]
        break
    
    k_lower = k_list[id - 1]
    k_upper = k_list[id + 1]
    new_range = k_upper - k_lower + 1
    delta = new_range // 4 
    if delta == 1:
        k_lower = max(k_lower - 2, 1)
        k_upper = min(k_upper + 2, N)
    
min_k
# %%


# %%
    def k_by_cv(self, cv_option = 'brute', verbose = False):
        '''
        Find the optimal parameter k from cross-validation.
        '''
        if cv_option not in ['pd', 'brute', 'iter']:
            raise Exception("No such cv_option.")
        
        N = self.N
        score = []
        if cv_option == 'pd':
            k = 1
            while k <= N:
                if not self.__is_pd(k):
                    break
                score.append(self.__loss_func(k))
                k += 1
            ans_k = np.array(score).argmin() + 1
        elif cv_option == 'brute':
            score = [self.__loss_func(k) for k in range(1, N + 1)]
            ans_k = np.array(score).argmin() + 1
        elif cv_option == 'iter':
            '''
            Initialize delta = N/4
            Search k-1 = {0, delta, 2*delta, ...}, 
            Then we can locate the minimum in [k*delta, (k+2)*delta], 
            and delta = delta/4, repeat the previous steps
            '''
            k_lower = 1
            k_upper = N
            delta = N // 4 + 1
            
        
        if verbose:
            print(score)
            plt.plot(score)
            plt.show()
            
        return ans_k