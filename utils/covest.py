# %%
from wlpy.gist import heatmap, generalized_threshold
from wlpy.covariance import Covariance
from sklearn import covariance as sk_cov
import numpy as np
from scipy import linalg as LA
import matplotlib.pyplot as plt
import pandas as pd
from wlpy.dataframe import DFManipulation
# %%


class CovEstWithNetwork(Covariance):
    """
    The model for estimating covariance using network information.\n
    Input: a T*N dataframe X \n
    Output: estimations of the N*N covariance using different methods.\n
    """

    def __init__(self, DF, *output_path):
        self.X = DF
        self.T, self.N = DF.shape
        self.I = np.eye(self.N)
        self.output_path = '/Users/lwg342/Documents/GitHub/Site-Generation/content/Covariance Estimation with Auxiliary Information/'
        # print('Edges in G', G.sum())
        
        self.scaling_factor = np.sqrt(np.log(self.N) / self.T)

    def fit(self, G, option='adaptive correlation threshold'):
        """
        Use the DF and the additional network matrix G to generate the estimate
        option: 1. 'projection' 2. 'lw2003' 3. 'adaptive correlation threshold' 
        """
        if option == 'adaptive correlation threshold':
            params = {'G': G, 'method': 'probit'}
        elif option == 'linear shrinkage with projection':
            self.fit_with_projection_method()
        else:
            print('No such option')

    """
    The following is the 1st attempt
    Find the estimate with projection method. 
    It can accommodate multiple targets but is hard to compute
    """

    def ip(self, A, B):
        result = np.trace(A@B.transpose())/self.N
        return result

    def ip_list(self, List_of_Matrices):
        result = [[self.ip(i, j) for i in List_of_Matrices]
                  for j in List_of_Matrices]
        return result

    def beta_sqr(self, A, AX):
        A = sk_cov.EmpiricalCovariance().fit(AX).covariance_
        _sum = 0
        for j in range(0, self.T):
            _sum = _sum + self.ip(np.outer(AX[j, :], AX[j, :]) - A,
                                  np.outer(AX[j, :], AX[j, :]) - A)
        beta_sqr = _sum/self.T/self.T
        return beta_sqr

    def delta_sqr(self, A):
        _mu = self.ip(A, self.I)
        delta_sqr = self.ip(A - _mu*self.I, A - _mu*self.I)
        return delta_sqr

    def alpha_sqr(self, A, AX, option='sample'):
        if option == 'sample':
            alpha_sqr = self.delta_sqr(A) - self.beta_sqr(A, AX)
        # if the second observation is fixed or independent of sample covariance.
        elif option == 'other':
            alpha_sqr = self.ip(A, self.S_sample)
        return alpha_sqr

    def fit_with_projection_method(self, S_list, option='s'):
        # We take S0, sample cov as the baseline, we add other matrices as targets
        S1 = self.S_sample
        mu = [self.ip(S1, self.I)]
        mu = mu + [self.ip(x, self.I) for x in S_list]
        dS1 = S1 - mu[0]*self.I
        # dS2 = S2 - mu2*self.I
        M = self.ip_list(([S1] + S_list))
        alpha = [self.alpha_sqr(S1, self.X)]
        alpha = alpha + [self.alpha_sqr(x, self.X, 'other') for x in S_list]
        params = LA.inv(M)@alpha
        S_new = np.zeros(self.N)
        for j in range(len(params)):
            S_new = S_new + params[j] * ([S1] + S_list)[j]
        if option == 'print':
            # print(LA.norm(S_new - S, ord=1))
            print('parameters are', params, '\n\n')
            print('mu', mu, '\n\n')
            print(np.array(M), '\n\n')
            print('alpha', alpha, '\n\n')
            print('Error:', LA.norm(S_new - self.S))
        self.S_new = S_new
        if option == 's' or option == 'print':
            return S_new
        elif option == 'p':
            return params

    """
    Fit with a single target, uses Ledoit and Wolf 2003 method. 
    Seems to have some errors. Incomplete
    """

    def fit_single_target(self, S_target, G=None):  # option = 'network'):
        """
        This is from schafer2005ShrinkageApproach
        If G is None, then we are using a constant targeting matrix
        If G is given, then we are using sample covariance with hard thresholding
        """
        X = self.X
        X_demean = X - X.mean(0)
        S_sample = self.sample_cov()
        denominator = ((S_sample - S_target) ** 2).sum()
        if np.any(G == None):
            print('unfinished')
        else:
            W = np.array([np.outer(x, x)
                          for x in X_demean])
            W_d = W - W.mean(0)
            variance_of_s = (self.T * (self.T - 1)**-3) * (W_d**2).sum(0)
            variance_of_s[np.where(G + np.eye(self.N) == 0)] = 0
            numerator = variance_of_s.sum()
            optimal_weight = numerator / denominator
            optimal_weight = max(min(optimal_weight, 1), 0)
            S_shrink_single_target = optimal_weight * \
                S_target + (1 - optimal_weight) * S_sample
            self.optimal_weight = optimal_weight
            self.S_shrink_single_target = S_shrink_single_target
            return S_shrink_single_target

    """
    option: 'adaptive correlation threshold'
    We apply generalized thresholding to the correlation matrix
    The threshold parameter uses network information
    """
    def feed_network(self, DF_G):
        self.G = DF_G
        return self

    def correlation_threshold_level(self,DF_G, tau_method='probit', params =None):
        """
        This generates a matrix mat_threshold that contains the individual threshold values
        if tau_method = 'direct': use the value in DF_G 
        if tau_method = 'probit': use probit model
        """
        N = self.N
        scaling_factor = self.scaling_factor 
        if tau_method == 'direct':
            endog = [DF_G.to_numpy().flatten()]
            intercept_term = np.ones(endog[0].shape)
            X = np.array([intercept_term] + endog)
            _Tau = (X.transpose()@ params).reshape([N, N]) *scaling_factor
            return _Tau
        elif tau_method == 'probit':
            from wlpy.gist import linear_probit
            _Tau = linear_probit([DF_G.to_numpy().flatten()],
                                 [params[0], - params[1]], add_constant=1).reshape([N, N]) * scaling_factor
            return _Tau
        elif tau_method == 'linear':
            endog = [DF_G.to_numpy().flatten()]
            intercept_term = np.ones(endog[0].shape)
            X = np.array([intercept_term] + endog)
            _Tau = (X.transpose()@params).reshape([N,N]) * scaling_factor
            return _Tau


    def fit_adaptive_corr_threshold(self, DF_G = None,  tau_method='linear', threshold_method='soft threshold', params = None,show_matrices = False, R = None, **kwargs):
        """
        This function will give the estimate\n
        """
        from wlpy.gist import generalized_threshold
        N = self.N
        X = self.X
        DF_G = self.G
        if R == None:
            R = np.abs(np.array(X.corr()))
        
        Tau = self.correlation_threshold_level(DF_G, tau_method, params)
        if show_matrices:
            print(Tau[:3, :3], '\n\n')

        _R_est = generalized_threshold(R, Tau, threshold_method)
        if show_matrices:
            print(_R_est[:3,:3], '\n\n')
        _R_est = _R_est - np.diag(np.diag(_R_est)) + np.eye(N)
        _S_est = np.diag(np.diag(self.sample_cov()) **
                         0.5) @ _R_est @ np.diag(np.diag(self.sample_cov())**0.5)
            # print(_S_est[:3,:3])
        self.S_adpt_corr = _S_est
        self.Tau_min_max = np.array([np.min(Tau), np.max(Tau)])
        return _S_est

    def loss_func(self, params, tau_method='linear',  g_thresholding_method = 'soft threshold'):
        from sklearn.model_selection import train_test_split
        DF_G = self.G
        V = 2
        test_size = 0.4
        score = np.zeros(V)
        for v in range(V):
            A, B = train_test_split(self.X, test_size=test_size)
            S_train = CovEstWithNetwork(A).fit_adaptive_corr_threshold(DF_G, tau_method, g_thresholding_method , params)
            S_validation = CovEstWithNetwork(B).sample_cov()
            score[v] = LA.norm(S_train - S_validation)**2
        average_score = score.mean()
        return average_score

    def find_smallest_threshold_for_pd(self, eig_id=np.array([0]), threshold_method = 'soft threshold', verbose=False):
    # print('\n\nNotice that here we use linear tau_method to find the minimum threshold level\n\n')
        smallest_eigenvalue = []
        threshold_level = []
        parameter_range = np.linspace( 0 , int(np.floor(1/self.scaling_factor)) + 1, 100)
        for c in parameter_range:
            self.fit_adaptive_corr_threshold(
                self.G, tau_method='linear', threshold_method=threshold_method, params=[c, 0])
            smallest_eigenvalue = smallest_eigenvalue + \
                [LA.eigvalsh(self.S_adpt_corr)[eig_id]]
            threshold_level = threshold_level + [self.Tau_min_max]
        result_df = pd.DataFrame(smallest_eigenvalue)
        result_df[['min', 'max']] = pd.DataFrame(threshold_level)
        result_df.min_tau_pd = result_df.loc[result_df[0] > 0]['min'].min()
        if verbose:
            print(result_df)
            fig, ax = plt.subplots()
            ax.plot(parameter_range, smallest_eigenvalue)
            fig.show()
            return result_df
        else:
            return result_df.min_tau_pd

    def params_by_cv(self, option = '', b = 0, **kwargs):
        """
        Notice: Here I have taken the constraint to be that the threshold is no less than 0.3, which comes from analysis of smallest eigenvalues that guarantee pd for the whole sample. Need to improve it to be data-driven.
        Find the optimal parameters from cross-validation method\n

        options:\n
        \t - 'brute' : naive brute force\n
        \t - 'pd': minimization with range constraints determined to guarantee pd
        """

        from scipy import optimize
        if option == 'brute':
            b_range = slice(-3, 0, 0.5) 
            a_range = slice(-2, 2, 0.5)
            rranges = (a_range,b_range)
            result = optimize.brute(self.loss_func, rranges)
        elif option == 'pd' and kwargs['tau_method'] == 'linear':
            scaling_factor = self.scaling_factor
            lb = [b,0,-1] / scaling_factor
            ub = [1,1,0] / scaling_factor
            linear_constraint = optimize.LinearConstraint([[1, 1], [1,0], [0,1]], lb , ub)
            x0 = [0,0]
            result = optimize.minimize(self.loss_func, x0, method='trust-constr', constraints=[linear_constraint],options={'verbose': 1}).x
        else: 
            result = None
            print('No result, check arguments')
        return result

    """
    Some common functions for generating reports\n
    """

    def _get_dict(self):
        _dict = {
            'Observation List': self.X,
            'Population Covariance': self.S,
            'Sample Cov': self.S_sample,
            'Linear Shrinked': self.S_lw,
        }
        return _dict

    def save_figures(self):
        heatmap(self.S, 'Population covariance matrix Sigma', self.output_path)
        heatmap(self.S_sample, 'Sample covariance matrix', self.output_path)
        heatmap(self.S_lw, 'Ledoit-Wolf Shrinkage Estimate of Sigma', self.o)
        heatmap(self.S_hard_threshold,
                'Hard Thresholding with DF_G Estimate of Sigma', self.output_path)
        heatmap(self.S_new, 'New srinkaged estimate', self.output_path)


# %%

class RetDF(DFManipulation):
    """
    We manipulate the Return Dataframe \n
    DF are T*N \n
    """
    def __init__(self, RET, FACTOR, RF):
        self.DF = RET
        self.RET = RET
        self.FACTOR = FACTOR.loc[RET.index]
        self.RF = RF.loc[RET.index]
        self._excess_return()
        self._defactor_excess_return()
        self._factor_covariance()

    def _excess_return(self):
        """
        Make sure that FACTOR has a column of risk-free rate with name 'rf'\n
        k is the number of factors to use 
        """
        RET = self.RET
        RF = self.RF
        excess_return = RET.sub(RF, axis=0)
        self.excess_return = excess_return
        # heatmap(excess_return.corr())
        # sns.distplot(excess_return.corr().to_numpy().flatten(), kde = True)
        # plt.show()
    
    def _defactor_excess_return(self):
        from wlpy.regression import OLS
        self.beta = OLS(self.FACTOR, self.excess_return).beta_hat()
        self.factor_component = OLS(self.FACTOR, self.excess_return).y_hat()
        self.excess_return_defactor = self.excess_return.sub(self.factor_component)
        
    def _factor_covariance(self):
        self.Sigma_factor = self.FACTOR.cov()
        self.BSigma_factorB = self.beta[self.FACTOR.columns] @ self.Sigma_factor @ self.beta[self.FACTOR.columns].transpose()
        return self.BSigma_factorB
    
# %% 


class NetBanding(Covariance):
    def __init__(self, X: np.array, G: np.array, N: int = None, T: int = None, threshold_method='soft threshold', cv_bound=0, use_correlation= True,  **kwargs):
        """
        Assume we observe N individuals over T periods, we want to use Network G guided banding method to obtain an N*N estimate of the assumed sparse covariance. 
        """
        super().__init__(X)
        self.X = X
        self.G = G
        if N == None and T ==None:
            T,N = X.shape
        self.N = N 
        self.T = T
        self.threshold_method = threshold_method
        self.scaling_factor = np.sqrt(np.log(self.N) / self.T)
        self.use_correlation = use_correlation
        
        self.sample_std_diagonal= np.diag(np.diag(self.sample_cov()))**0.5
        self.R = np.corrcoef(np.array(self.X), rowvar= False)

        
    def fit(self, params, option = None):
        if self.use_correlation:
            M = self.R
            # print("correlation!!!")
        else:
            M = self.S_sample
            
        if option == "universal":
            G = np.zeros([self.N, self.N])
        else:
            G = self.G
        M1 = np.where(G==1, M, 0)
        M0 = np.where(G ==0 , M, 0)
        Tau = params * np.ones([self.N, self.N]) * self.scaling_factor
        M0T = generalized_threshold(M0, Tau, self.threshold_method)
        M_new = M1 + M0T
        if self.use_correlation:
            M_new = M_new - np.diag(np.diag(M_new)) + np.eye(self.N)
            self.Rt = M_new
            S_new = self.sample_std_diagonal @ M_new @ self.sample_std_diagonal
        else:
            S_new = M_new
        return S_new
    
    def loss_func(self, params):
        from sklearn.model_selection import train_test_split
        # V = self.num_cv
        test_size = int(np.floor(self.T * 0.4))
        V= 4
        score = np.zeros(V)
  
        for v in range(V):
            A, B = train_test_split(self.X, test_size= test_size)
            S_train = NetBanding(A, self.G, self.N, self.T).fit(params)
            S_validation = np.cov(np.array(B), rowvar= False)
            score[v] = LA.norm(S_train - S_validation)**2
        average_score = score.mean()
        
        return average_score
    
    def params_by_cv(self, cv_option = "brute", cv_bound = 0, **kwargs):
        """
        Find the optimal parameters from cross-validation method\n

        options:\n
        \t - "brute" : naive brute force\n
        \t - "pd": minimization with range constraints determined to guarantee pd
        """

        from scipy import optimize
        if cv_option == "brute":
            result = optimize.brute(
                self.loss_func, (slice(0, 1.0, 0.1),))
        elif cv_option == "pd":
            scaling_factor = self.scaling_factor
            lb = cv_bound/ scaling_factor
            ub = 1/ scaling_factor
            x0 = np.array([0])
            result = optimize.minimize(
                self.loss_func, x0, method="trust-constr", bounds=((lb, ub),), options={"verbose": 0}).x
        else: 
            result = None
            print("No result, check arguments")
        return result
        

# %%

