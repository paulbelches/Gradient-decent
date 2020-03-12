import numpy as np

def linear_cost(X, y, theta, lambd):
    m, _ = X.shape
    h = np.matmul(X, theta)
    sq = (y - h) ** 2
    theta = lambd * (theta ** 2)#
    return (theta.sum() + sq.sum()) / (2 * m)#
    #return sq.sum() / (2 * m)
