# %%
import numpy as np
import warnings
import matplotlib.pyplot as plt
from scipy import linalg as LA
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
class InfoCorrBand():
    '''
    Steps to get the estimator:
    1. feed the object with auxiliary information L-matrix
    2. use cross-validation to determine k, i.e. params
    3. construct the estimator with k
    '''
    
    def __init__(self, X, L = None,  
                 num_cv = 4, test_size = None):
        '''
        X : 2-D array 
            Data. Each row is an observation. Each column is a random variable. Of size T*N.
            
        L : 2-D array 
            Symmetric auxiliary information matrix of size N*N. Each entry is a positive float, whose magnitude resembles the magnitude of the corresponding correlation coefficient.
        
        num_cv : int, optional
            Repeat cross-validation for num_cv times before computing the average score.
        
        test_size : int, optional
            The fraction of observations in the test set in cross validation.
        '''
        self.X = X 
        self.T, self.N = X.shape
        # self.L and self.rowSort are initialized in feed_info().
        self.L = None
        self.rowSort = None
        self.feed_info(L)
        self.I = np.eye(self.N)
        '''
        # NO NEED. 
        # uniformity class parameter. 
        self.q = q  
        self.alpha = alpha
        self.scaling_factor = (np.log(self.N) / self.T) ** (-q / (2*alpha + 2))
        '''
        self.num_cv = num_cv
        if test_size is None:
            self.test_size = 2 / 3 # 1 / np.log(self.T)
        else:
            self.test_size = test_size
        self.eps = 1e-10
        self.D_est = np.diag(np.diag(self.sample_cov())) ** 0.5
        
    def feed_info(self, L = None):
        '''
        This func is first called when __init__().
        You can also change L-matrix later on.
        '''
        if L is None:
            warnings.warn('L-matrix is missing.')
            return 
        if self.N != L.shape[0]:
            raise Exception('The sizes of X and L do not match.')
        if not (L == L.transpose()).all(): # asymmetric
            warnings.warn('Input L-matrix is asymmetric.', DeprecationWarning)
        self.L = (L + L.transpose()) / 2
        self.__compute_orders()
        
    def __compute_orders(self):
        '''
        A private function called by feed_info().
        Generates rowSort-matrix.
        rowSort : 2-D array
            'self.rowSort[i,j]=k' means in row/column i, self.L indicates that j is the k-biggest in magnitude
        '''
        L = self.L
        N = self.N
        rowSort = np.zeros((N, N))
        for i in range(N):
            f = lambda x: np.argsort(x)
            rowSort[i] = N - f(f(L[i]))
            '''
            Order = np.argsort(L[i])[::-1]
            # tmp = np.zeros(N) # does accelerate 44%
            for k in range(N):
                # tmp[Order[k]] = k+1 #
                rowSort[i][Order[k]] = k+1
            # N - np.argsort(np.argsort(L[i])) == rowSort[i])), it has been tested
            # rowSort[i] = tmp #
            '''
        self.rowSort = rowSort
        
    def sample_cov(self):
        return np.cov(self.X, rowvar = False)
    
    def sample_corr(self) -> np.ndarray:
        return np.corrcoef(self.X, rowvar = False)
    
    """
    def find_biggest_k_for_pd(self):
        '''
        When k=N, the estimator reduces to the sample covariance, not P.D.
        When k=1, the estimator becomes diagonal, P.D.
        '''
        left, right = 1, self.N
        '''
        Binary search for the biggest k with P.D. estimator.
        Search in the interval [left, right]
        '''
        while left < right: 
            mid = (left + right) // 2 + 1
            EigVals = np.linalg.eigvals(self.fit_info_corr_band(k = mid))
            if EigVals[-1] > self.eps:
                left = mid
            else:
                right = mid - 1
        return mid
    """
    
    def k_by_cv(self, cv_option = 'fast_iter', verbose = False):
        '''
        Find the optimal parameter k from cross-validation.
        '''
        if cv_option not in ['pd', 'brute', 'fast_iter']:
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
        if verbose:
            print(score)
            plt.plot(score)
            plt.show()
        
        if cv_option == 'fast_iter':
            '''
            This algorithm ~ log(N)
            1. Initialize delta = N/4, interval = [1, N]
            2. compute the score when k-1 in S = {0, delta, 2*delta, ...}
            3. Find m, such that m*delta corresponds to the smallest score.
            4. delta = delta/4, interval = [(m-1)*delta, (m+1)*delta], repeat step 2 and step 3 until delta = 1
            5. expand interval a little bit, e.g., [low, up] -> [low - 2, up + 2], brute force search
            '''
            score_dict = dict()
            def tmp__loss_func(k):
                if k not in score_dict:
                    score_dict[k] = self.__loss_func(k)
                return score_dict[k]
                    
            k_lower = 1
            k_upper = N
            delta = N // 4
            
            
            my_arr = [tmp__loss_func(k) for k in range(1, 301, 2)]
            my_id = range(1, 301, 2)
            plt.plot(my_id, my_arr)
            plt.show()
            
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
            
        return ans_k
        
    def __loss_func(self, k):
        from sklearn.model_selection import train_test_split
        v = self.num_cv
        score_i = np.zeros(v)
        for i in range(v):
            X1, X2 = train_test_split(self.X, test_size = self.test_size, random_state = i) # test_size = proportion of X2
            o1 = InfoCorrBand(X1, self.L)
            o2 = InfoCorrBand(X2) # needn't to call  __compute_orders
            R_est1 = o1.fit_info_corr_band(k)
            R_est2 = o2.sample_corr()
            score_i[i] = LA.norm(R_est1 - R_est2) ** 2
        return score_i.mean()
    
    def fit_info_corr_band(self, k):
        R_est = self.sample_corr()
        N = self.N
        rS = self.rowSort
        if rS is None:
            raise Exception("Please call InfoCorrBand.feed_info() function first.")
        Taper = ((rS <= k) & (rS.T <= k)).astype(int)
        return R_est * Taper # Hadamard product
        '''
        for i in range(N):
            for j in range(N):
                if not (rowSort[i][j] <= k and rowSort[j][i] <= k):
                    R_est[i][j] = 0
        return R_est
        '''
    
    def fit_info_cov_band(self, k):
        return self.D_est @ self.fit_info_corr_band(k) @ self.D_est
    
    def plot_k_pd(self, k_range = None):
        '''
        Plot 'k'-'smallest eigenvalue'.
        '''
        N = self.N
        if k_range is None:
            k_range = range(N - N // 4, N + 1)
        vals = [np.linalg.eigvals(self.fit_info_corr_band(k))[-1] 
                for k in k_range]
        plt.plot(k_range, vals)
        plt.xlabel('parameter k')
        plt.ylabel('smallest eigenvalue of correlation estimator')
        # plt.savefig('../figs/plot_k_pd.jpg')
        plt.show()
        return
    
    def __is_pd(self, k) -> bool:
        return np.linalg.eigvals(self.fit_info_corr_band(k))[-1] > self.eps
    
    def auto_fit(self):
        k = self.k_by_cv()
        R_est = self.fit_info_corr_band(k)
        S_est = self.D_est @ R_est @ self.D_est
        return R_est, S_est, k
    
    # def params_by_cv(self)
# %%