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
class InfoCorrBand():
    '''
    Steps to get the estimator:
    1. feed the object with auxiliary information L-matrix
    2. use cross-validation to determine k, i.e. params
    3. construct the estimator with k
    '''
    
    def __init__(self, X, L = None,  
                 num_cv = 4, test_size = 0.4):
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
        self.test_size = test_size
        self.eps = 1e-10
        
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
        self.L = (L + L.transpose()) /2
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
            Order = np.argsort(L[i])[::-1]
            for k in range(N):
                rowSort[i][Order[k]] = k+1
        self.rowSort = rowSort
        
    def sample_cov(self):
        return np.cov(self.X, rowvar = False)
    
    def sample_corr(self):
        return np.corrcoef(self.X, rowvar = False)
    
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
    
    def k_by_cv(self, cv_option = 'pd', verbose = False):
        '''
        Find the optimal parameter k from cross-validation.
        '''
        N = self.N
        score = []
        if cv_option == 'pd':
            k = 1
            while k <= N:
                if not self.__is_pd(k):
                    break
                score.append(self.__loss_func(k))
                k += 1
            score = np.array(score)
            if verbose:
                print(score)
            return score.argmin() + 1
        raise Exception("Now we only have 'pd' cross-validation option.")
        
    def __loss_func(self, k):
        from sklearn.model_selection import train_test_split
        v = self.num_cv
        score_i = np.zeros(v)
        for i in range(v):
            X1, X2 = train_test_split(self.X, test_size = self.test_size)
            o1 = InfoCorrBand(X1, self.L)
            o2 = InfoCorrBand(X2, self.L)
            R_est1 = o1.fit_info_corr_band(k)
            R_est2 = o2.sample_corr()
            score_i[i] = LA.norm(R_est1 - R_est2) ** 2
        return score_i.mean()
    
    def fit_info_corr_band(self, k):
        R_est = self.sample_corr()
        N = self.N
        rowSort = self.rowSort
        if rowSort is None:
            raise Exception("Please call InfoCorrBand.feed_info() function first.")
        for i in range(N):
            for j in range(N):
                if not (rowSort[i][j] <= k and rowSort[j][i] <= k):
                    R_est[i][j] = 0
        return R_est
    
    def fit_info_cov_band(self, k):
        Sigma_est = self.sample_cov() # sample covariance
        D_est = np.sqrt(np.diag(np.diag(Sigma_est))) # sample marginal deviations
        return D_est @ self.fit_info_corr_band(k) @ D_est
    
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
        plt.savefig('../figs/plot_k_pd.jpg')
        plt.show()
        return
    
    def __is_pd(self, k) -> bool:
        return np.linalg.eigvals(self.fit_info_corr_band(k))[-1] > self.eps
    
    def auto_fit(self):
        k = self.k_by_cv()
        return self.fit_info_corr_band(k), self.fit_info_cov_band(k), k
    
    # def params_by_cv(self)
# %%