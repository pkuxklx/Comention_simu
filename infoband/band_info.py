# %%
import numpy as np
import warnings
import matplotlib.pyplot as plt
from scipy import linalg as LA
from wlpy.covariance import Covariance
# %%
import sys 
sys.path.append("..")
from utils.adpt_correlation_threshold import AdptCorrThreshold
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
class InfoCorrBand(Covariance):
    '''
    Steps to get the estimator:
    1. feed the object with auxiliary information L-matrix
    2. use cross-validation to determine k, i.e. params
    3. construct the estimator with k
    '''
    
    def __init__(self, X: np.array, L: np.array = None, num_cv = 10, test_size: float = None):
        """
        Args:
            X: 2-D array Data. Each row is an observation. Each column is a random variable. Of size T*N.
            
            L: 2-D array Symmetric auxiliary information matrix of size N * N. Each entry is a positive float, whose magnitude resembles the magnitude of the corresponding correlation coefficient.
        
            num_cv: int, optional. Do random split for num_cv times in cross-validation.
        
            test_size: float, optional. The fraction of observations in the test set in CV.
        """
        super().__init__(X)
        # self.L and self.rowSort are initialized in feed_info().
        self.L = None
        self.rowSort = None
        self.feed_info(L)
        self.num_cv = num_cv
        self.test_size = 1 / np.log(self.T) if test_size is None else test_size
        
    def feed_info(self, L = None):
        """
        Get L-matrix. This func is first called when __init__(). You can also change L-matrix later on.
        """
        if L is None:
            warnings.warn('L-matrix is missing.')
            return 
        if self.N != L.shape[0]:
            raise Exception('The sizes of X and L do not match.')
        if not (L.argmax(axis = 0) == np.arange(self.N)).all():
            warnings.warn("In each row of L-matrix, the diagonal element should be treated as the most important.")
            assert (L >= 0).all()
            m = L.max()
            L = L + np.eye(self.N) * (m + 1)
        if not (L == L.transpose()).all(): # asymmetric
            warnings.warn('Input L-matrix is asymmetric.', DeprecationWarning)
            L = (L + L.transpose()) / 2
        self.L = L
        self.__compute_orders()
        
    def __compute_orders(self):
        """
        A private function called by feed_info(). Generates rowSort-matrix.
        rowSort: 2-D array
            'self.rowSort[i,j]=k' means in row/column i, self.L indicates that j is the k-biggest in magnitude
        """
        L = self.L
        N = self.N
        rowSort = np.zeros((N, N))
        for i in range(N):
            f = lambda x: np.argsort(x)
            rowSort[i] = N - f(f(L[i])) # argsort for 2 times: (b,a,c) => (2,1,3) if a<b<c
        self.rowSort = rowSort
    
    def params_by_cv(self, cv_option = 'brute', **kwargs):
        """
        Find the optimal parameter k by cross-validation.
        """
        if cv_option not in ['pd', 'brute', 'fast_iter']:
            raise ValueError("Invalid cv_option.")
        
        N = self.N
        score = []
        if cv_option == 'pd':
            k = 1
            while k <= N:
                if np.linalg.eigvalsh(self.fit([k], ad_option = None)).min() <= 0: # not PD
                    break
                score.append(self.loss_func([k]))
                k += 1
            ans_k = np.array(score).argmin() + 1
        elif cv_option == 'brute':
            score = [self.loss_func([k]) for k in range(1, N + 1)]
            ans_k = np.array(score).argmin() + 1
        elif cv_option == 'fast_iter':
            warnings.warn('Not robust.', DeprecationWarning)
            '''
            This algorithm is an adapted version of the ternary search algorithm for the minimum of a U-shaped curve. 
            The time complexity ~ log(N).
            1. Initialize delta = N/4, interval = [1, N]
            2. compute the score when k-1 in S = {0, delta, 2*delta, ...}
            3. Find m, such that m*delta corresponds to the smallest score.
            4. delta = delta/4, interval = [(m-1)*delta, (m+1)*delta], 
                then repeat step 2 and step 3 until delta <= threshold_value.
            5. expand interval a little bit, e.g., [low, up] -> [low - 2, up + 2], then brute force search.
            '''
            score_dict = dict()
            def tmp__loss_func(k):
                if k not in score_dict:
                    score_dict[k] = self.loss_func([k])
                return score_dict[k]
                    
            k_lower = 1
            k_upper = N
            delta = N // 4
            
            while 1:
                k = k_lower
                k_list, k_score = [], []
                while k <= k_upper:
                    k_score.append(tmp__loss_func(k))
                    k_list.append(k)
                    k = k + 1 if k == k_upper else min(k + delta, k_upper)
                id = np.array(k_score).argmin()
                if delta == 1:
                    # use MA, more robust
                    MA_k_score = moving_average(k_score)
                    MA_id = np.array(MA_k_score).argmin()
                    ans_k = k_list[MA_id]
                    plt.figure(figsize = (4, 2))
                    plt.subplot(1, 2, 1)
                    plt.plot(k_list, k_score)
                    plt.subplot(1, 2, 2)
                    plt.plot(k_list, MA_k_score)
                    plt.show()
                    break
                
                # the range of next iteration, is determined by this iteration's minimum position 'id'
                k_lower = k_list[max(id - 1, 0)]
                k_upper = k_list[min(id + 1, len(k_list) - 1)]
                new_range = k_upper - k_lower + 1
                delta = new_range // 4 
                if delta <= 3:
                    delta = 1
                    k_lower = max(k_lower - 2, 1)
                    k_upper = min(k_upper + 2, N)
            
        return np.array([ans_k])
        
    def loss_func(self, params):
        k = params[0]
        from sklearn.model_selection import train_test_split
        V = self.num_cv
        score = np.zeros(V)
        for v in range(V):
            X1, X2 = train_test_split(self.X, test_size = self.test_size, random_state = v)
            o1 = InfoCorrBand(X1, self.L)
            o2 = InfoCorrBand(X2) # needn't to call  __compute_orders
            R_train = o1.fit(params, ad_option = None, ret_cor = True) # when cv, we don't do modifications (ad_option = None), and use correlation matrices.
            R_validation = o2.R_sample
            score[v] = LA.norm(R_train - R_validation) ** 2

        return score.mean()
    
    def fit(self, params, ad_option = None, ret_cor = False, eps = 1e-4, **kwargs):
        """
        Args:
            params: [k].
    
            ad_option: Adjustment after fitting.
    
            ret_cor: If True, return the correlation estimator. 

            eps: Only used when ad_option = 'pd'. The smallest eigenvalue.
        """
        k = params[0]
        
        N = self.N
        rS = self.rowSort
        if rS is None:
            raise Exception("Please call InfoCorrBand.feed_info() function first.")
        Taper = ((rS <= k) & (rS.T <= k)).astype(int)
        R_est = self.R_sample * Taper 
        S_est = self.D_sample @ R_est @ self.D_sample

        if ad_option == 'pd':
            # Chen, 2019, A New Semiparametric Estimation Approach for Large Dynamic Covariance Matrices with Multiple Conditioning Variables
            w, V = np.linalg.eigh(S_est)
            w = w.clip(min = eps)
            S_est = V @ np.diag(w) @ V.T
        
        return R_est if ret_cor else S_est
    
    def plot(self, ranges = (slice(1, 50, None),), y_type = 'loss'):
        """
        This function facilitates debugging. Plot the tuning parameter against the loss or the smallest eigenvalue. 
        Args:
            ranges: tuple. Each element in the tuple is the range of one tuning parameter. In this case, there is only one param 'k'.
            y_type: str. 
        """
        arr = range(ranges[0].start, ranges[0].stop + 1)
        if y_type == 'loss':
            y = [self.loss_func([x]) for x in arr]
        elif y_type == 'eigval':
            y = [np.linalg.eigvalsh(self.fit([x], ad_option = None)).min() for x in arr]
        else:
            raise ValueError('Invalid y_type.')
        plt.plot(arr, y)
        plt.title(y_type)
        plt.xlabel('k')
        plt.show()

# %%