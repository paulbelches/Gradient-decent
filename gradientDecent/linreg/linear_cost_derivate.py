import numpy as np

def linear_cost_derivate(X, y, theta,lambd):
    h = np.matmul(X, theta)
    m, _ = X.shape
    return (np.matmul((h - y).T, X).T + (lambd * theta)) / m
    #return np.matmul((h - y).T, X).T / m
