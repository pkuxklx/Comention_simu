We find that the error rate of Augmented Thresholding (augment before threshold) is significantly more volatile than Hard Thresholding. 

One (The main ?) reason is that if we augment before choosing the threshold parameter by cross validation, the sample covariances are so unstable that the threshold parameter will have a high variance as well. 

Settings: 
* $N = 500, T = 300, \tau = 0.2, p=1, q=0, num\_cv=50$. 
* True covariance = gen_S_Cai2011Adaptive_Model2_my(N, seed = 0)
* Frobenius norm.