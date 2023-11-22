import numpy as np
from sklearn.neural_network import MLPClassifier, MLPRegressor
from scipy.special import erf

from numpy.linalg import cholesky, det
from scipy.linalg import solve_triangular
from scipy.optimize import minimize
from numpy.linalg import inv
import numpy as np

import matplotlib.pyplot as plt

def regres(X, y, xtest):
    regr = MLPRegressor(random_state=1, max_iter=500).fit(X, y)
    y_pred = regr.predict(xtest)
    #y_pred = y_mean - 1.96*std_pred, y_mean + 1.96*std_pred, 
    return y_pre

def classify(X, y, xtest):
    clf = MLPClassifier(random_state=1, max_iter=500).fit(X, y)
    y_pred = clf.predict(xtest)
    y_prob = clf.predict_proba(xtest)
    return y_pred, y_prob