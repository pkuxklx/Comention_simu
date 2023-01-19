from utils.covest import CovEstWithNetwork
import numpy as np
from wlpy.gist import heatmap
import matplotlib.pyplot as plt
from scipy import linalg as LA
import pandas as pd

from wlpy.report import Timer
tt = Timer()


class AdptCorrThreshold(CovEstWithNetwork): 
    # After looking up 'super()' in the script, I find there isn't any connection with the father class
    """
    option: "adaptive correlation threshold"
    We apply generalized thresholding to the correlation matrix
    The threshold parameter uses network information
    """

    def __init__(self, DF, DF_G, 
                 tau_method = "linear", threshold_method = "soft threshold", 
                 num_cv = 4, test_size = 0.4, cv_init_value = 0, 
                 cv_split_method = "random", split_point = [150], 
                 *output_path, **kwargs):
        super().__init__(DF)
        self.X = DF
        self.G = DF_G
        self.NPG = np.array(DF_G)
        self.T, self.N = DF.shape
        self.I = np.eye(self.N)
        self.output_path = "/Users/lwg342/Documents/GitHub/Site-Generation/content/Covariance Estimation with Auxiliary Information/"
        self.scaling_factor = np.sqrt(np.log(self.N) / self.T)
        self.sample_std_diagonal= np.diag(np.diag(self.sample_cov()))**0.5
        self.R = np.corrcoef(np.array(self.X), rowvar= False)
        self.R_sign = np.sign(self.R)
        
        self.tau_method= tau_method
        self.threshold_method = threshold_method
        self.num_cv = num_cv

        self.test_size = test_size
        self.cv_init_value = cv_init_value
        self.split_point = split_point
        self.cv_split_method = cv_split_method

    def feed_network(self, DF_G):
        self.G = DF_G
        return self

    def correlation_threshold_level(self, params =None):
        """
        This generates a matrix mat_threshold that contains the individual threshold values
        if tau_method = "direct": use the value in DF_G 
        if tau_method = "probit": use probit model
        """
        N = self.N
        DF_G = self.G
        scaling_factor = self.scaling_factor 

        if self.tau_method == "probit":
            from wlpy.gist import linear_probit
            _Tau = linear_probit([self.NPG.flatten()],
                                 [params[0], params[1]], add_constant=1).reshape([N, N]) * scaling_factor
            return _Tau

        elif self.tau_method == "old-linear":
            endog = [self.NPG.to_numpy().flatten()]
            intercept_term = np.ones(endog[0].shape)
            X = np.array([intercept_term] + endog)
            _Tau = (X.transpose()@params).reshape([N,N]) * scaling_factor
            return _Tau
        elif self.tau_method == "linear" and len(params) == 2: # default
            _Tau1 = (params[0] + params[1]*self.NPG) * scaling_factor
            return _Tau1
        elif self.tau_method == "simple" and len(params) == 1:
            _Tau1 = params * np.ones([self.N, self.N]) * scaling_factor
            return _Tau1
        else: 
            _Tau = None
            print("No such method")

    def fit_adaptive_corr_threshold(self, params = None, show_matrices = False, timer = False, **model_params):
        """
        This function will give the estimate\n
        """
        if timer:
            from wlpy.report import Timer
            tt = Timer()
        from wlpy.gist import generalized_threshold
        N = self.N
        R = self.R

        # Tau = self.correlation_threshold_level(params)
        Tau = self.correlation_threshold_level(params).clip(0)
        if show_matrices:
            print(Tau[:3, :3], "\n\n")

        _R_est = generalized_threshold(R, Tau, self.threshold_method, sign = self.R_sign)
        if show_matrices:
            print(_R_est[:3,:3], "\n\n")
        _R_est = _R_est - np.diag(np.diag(_R_est)) + np.eye(N)
        return _R_est
    
    def fit_adaptive_cov_threshold(self, params = None, show_matrices = False, timer = False, **model_params):
        R_est = self.fit_adaptive_corr_threshold(params, show_matrices, timer, model_params)
        S_est = (self.sample_std_diagonal) @ R_est @ (self.sample_std_diagonal)
        return S_est
    
    def auto_fit(self, threshold_method = None): 
        if threshold_method is not None: # change self.threshold_method
            self.threshold_method = threshold_method
        b = self.find_smallest_threshold_for_pd()
        params = self.params_by_cv('pd', b)
        R_est = self.fit_adaptive_corr_threshold(params)   
        S_est = self.sample_std_diagonal @ R_est @ self.sample_std_diagonal
        return R_est, S_est, params

    def loss_func(self, params):
        from sklearn.model_selection import train_test_split
        V = self.num_cv
        score = np.zeros(V)

        if self.cv_split_method == "sequential":
            v = 0
            for c in self.split_point:
                A = self.X.iloc[:c]
                B = self.X.iloc[c:]
                S_train = AdptCorrThreshold(
                    A, self.G, self.tau_method, self.threshold_method, sign=self.R_sign).fit_adaptive_corr_threshold(params)
                S_validation = np.cov(np.array(B), rowvar=False)
                score[v] = LA.norm(S_train - S_validation)**2 # frobenius norm
                v = v+1

        elif self.cv_split_method == "random": # default
            for v in range(V):
                # tt.start()
                A, B = train_test_split(self.X, test_size=self.test_size)
                S_train = AdptCorrThreshold(
                    A, self.G, self.tau_method, self.threshold_method, sign = self.R_sign).fit_adaptive_corr_threshold(params)
                S_validation = np.cov(np.array(B), rowvar= False)
                score[v] = LA.norm(S_train - S_validation)**2 
                # tt.click()
        average_score = score.mean()
        

        return average_score

    def find_smallest_threshold_for_pd(self, eig_id=np.array([0]),  verbose=False):
        """
        Notice that here we use linear tau_method to find the minimum threshold level
        """
        parameter_range = np.linspace( self.cv_init_value , int(np.floor(1/self.scaling_factor)) + 1, 100)
        if verbose:
            smallest_eigenvalue = []
            threshold_level = []
            for c in parameter_range:
                self.fit_adaptive_corr_threshold(params=[c, 0], tau_method = "linear")
                smallest_eigenvalue = smallest_eigenvalue + \
                    [LA.eigvalsh(self.S_adpt_corr)[eig_id]]
                threshold_level = threshold_level + [self.Tau_min_max]
            result_df = pd.DataFrame(smallest_eigenvalue)
            result_df[["min", "max"]] = pd.DataFrame(threshold_level)
            result_df.min_tau_pd = result_df.loc[result_df[0] > 0]["min"].min()
            print(result_df)
            fig, ax = plt.subplots()
            ax.plot(parameter_range, smallest_eigenvalue)
            fig.show()
            return result_df
        else:
            for c in parameter_range:
                S = self.fit_adaptive_corr_threshold(params=[c, 0])
                try:
                    np.linalg.cholesky(S)
                    # print(f"success with {c}*{self.scaling_factor}")
                    break
                except np.linalg.LinAlgError:
                    pass
                    # print(f"failure{c}")
            return c * self.scaling_factor

    def params_by_cv(self, cv_option = "pd", cv_bound = 0, **kwargs):
        """
        Find the optimal parameters from cross-validation method\n

        options:\n
        \t - "brute" : naive brute force\n
        \t - "pd": minimization with range constraints determined to guarantee pd
        """

        from scipy import optimize
        if cv_option == "brute":
            b_range = slice(-3, 0, 0.5) 
            a_range = slice(-2, 2, 0.5)
            rranges = (a_range,b_range)
            result = optimize.brute(self.loss_func, rranges)
        elif cv_option == "pd" and self.tau_method == "linear":
            scaling_factor = self.scaling_factor
            lb = [cv_bound,0,-1] / scaling_factor
            ub = [1,1,0] / scaling_factor
            linear_constraint = optimize.LinearConstraint([[1, 1], [1,0], [0,1]], lb , ub)
            x0 = [0,0]
            result = optimize.minimize(self.loss_func, x0, method="trust-constr", constraints=[linear_constraint],options={"verbose":0}).x
        else: 
            result = None
            print("No result, check arguments")
        return result

    # ------------------ # 
    def shrink_to_network_target():
        pass

    """
    Some common functions for generating reports\n
    """

    def _get_dict(self):
        _dict = {
            "Observation List": self.X,
            "Population Covariance": self.S,
            "Sample Cov": self.S_sample,
            "Linear Shrinked": self.S_lw,
        }
        return _dict

    def save_figures(self):
        heatmap(self.S, "Population covariance matrix Sigma", self.output_path)
        heatmap(self.S_sample, "Sample covariance matrix", self.output_path)
        heatmap(self.S_lw, "Ledoit-Wolf Shrinkage Estimate of Sigma", self.o)
        heatmap(self.S_hard_threshold,
                "Hard Thresholding with DF_G Estimate of Sigma", self.output_path)
        heatmap(self.S_new, "New srinkaged estimate", self.output_path)
