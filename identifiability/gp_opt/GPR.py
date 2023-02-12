import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF,ConstantKernel, Matern
from scipy.special import erf

from numpy.linalg import cholesky, det
from scipy.linalg import solve_triangular
from scipy.optimize import minimize
from numpy.linalg import inv
import numpy as np

import matplotlib.pyplot as plt

def regres(X, y, xtest):
    k1 = ConstantKernel(constant_value=1.0,constant_value_bounds=(1e-10,1e10))
    k2 = RBF(length_scale=0.1,length_scale_bounds=(1e-10,1e10))
    k3 = 1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-10,1e10), nu=1.5)
    k4 = ConstantKernel(constant_value=1.0,constant_value_bounds=(1e-10,1e10))
    
    normalize = True
    n_restarts = 10
    optimizer = 'fmin_l_bfgs_b'
    noise = 0.02

    kernel = k1*k2 + k4*k3
    gp = GaussianProcessRegressor(kernel=kernel, \
        n_restarts_optimizer=n_restarts, \
        #alpha = noise, \
        optimizer=optimizer)
    gp.fit(X, y)
    y_mean, std_pred = gp.predict(xtest, return_std=True)
    y_pred = y_mean - 1.96*std_pred, y_mean + 1.96*std_pred, 
    return y_mean, y_pred, std_pred