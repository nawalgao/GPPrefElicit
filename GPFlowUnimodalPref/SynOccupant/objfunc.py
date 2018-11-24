from scipy.stats import multivariate_normal, beta
import numpy as np

def u2D(X, m1, v1, m2, v2, off):
    """
    Synthetic thermal utility for 2D features
    Multivariate Gaussian Distribution
    Operating temp. and relative humidity
    """
    mean_vec = np.array([m1, m2])
    cov_mat = np.array([[v1, off],[off, v2]])
    normal = multivariate_normal(mean=mean_vec,
                                 cov = cov_mat)
    u_vec = normal.pdf(X)
    
    return u_vec 

def beta_utility_gen(X, x_max, a, b):
    """
    Beta distribution over utility value
    X : value of Vertical Illuminance (EV) at which to find the utility function value
    x_max : max value of EV to be considered for getting the proper Beta distribution
    a: Beta distribution parameter
    b : Beta distribution parameter
    Outputs:
    Utility function value
    """
    X = X - 15
    x_all = np.linspace(0,1,100)
    u_all = beta.pdf(x_all, a, b)
    u_max_all = np.max(u_all)
    u_min_all = np.min(u_all)


    x_norm = X/x_max
    u = beta.pdf(x_norm, a, b)
    u_der = ((a - 1)/x_norm - (b - 1)/(1 - x_norm))* u
    u_norm = (u - u_min_all)/(u_max_all - u_min_all)
    u_der_norm = u_der/(u_max_all - u_min_all)
    return u_norm, u_der_norm

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_context("talk", font_scale = 1.4)
    
    xx = np.linspace(20, 27, 200)
    x_max = 20
    a = 6
    b = 8
    u, u_der_norm = beta_utility_gen(xx, x_max, a, b)
    plt.figure(figsize = (12,8))
    plt.plot(xx, u, color = 'b', label = 'true utility', alpha = 0.75)
    plt.xlabel('Indoor Room Temperature [$^{o}$C]')
    plt.ylabel(' Normalized Utility $u_{n}$')
    plt.legend(loc = 'best')
    
