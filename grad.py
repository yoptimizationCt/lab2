import numpy as np


def grad(func, x_0):
    grad_epsilon = 1e-6
    n = len(x_0)  # dimension of х_0
    # [eps, eps, eps, ..., eps].size(n)
    ret = np.zeros(n)
    h = np.zeros(n)
    for i in range(n):
        h[i] = grad_epsilon
        ret[i] = (func(x_0 + h) - func(x_0)) / grad_epsilon
        h[i] = 0
    return ret
