import numpy as np

from paint_contour import paint_contour

n = 1000
X = np.random.rand(n) * 10
Y = 5 * X + np.random.rand(n) * 20


def gradient(point, j):
    return np.array(
        [2 * X[j] * (point[0] * X[j] + point[1] - Y[j]),
         2 * (point[0] * X[j] + point[1] - Y[j])])


def SGD(point=np.zeros(2), lr=0.0009, epoch_count=100):
    for epoch in range(1, epoch_count):
        j = np.random.randint(0, n)
        gr = gradient(point, j)
        point = point - lr * gr
    return point


def SGD_with_momentum(point=np.zeros(2), lr=0.0009, v=np.zeros(2), gamma=0.5, epoch_count=100):
    for epoch in range(1, epoch_count):
        j = np.random.randint(0, n)
        v = gamma * v + (1 - gamma) * gradient(point, j)
        point = point - lr * v
    return point


def Nesterov(point=np.zeros(2), lr=0.0009, v=np.zeros(2), gamma=0.5, epoch_count=100):
    for epoch in range(1, epoch_count):
        j = np.random.randint(0, n)
        v = gamma * v + (1 - gamma) * gradient(point - lr * gamma * v, j)
        point = point - lr * v
    return point


def RMS_prop(point=np.zeros(2), lr=0.0009, epoch_count=100, alpha=0.5, eps=10e-8):
    s = 0
    for epoch in range(1, epoch_count):
        j = np.random.randint(0, n)
        gr = gradient(point, j, X)
        s = alpha * s + (1 - alpha) * (gr * gr)
        point = point - lr * gr / (np.sqrt(s + eps))
    return point


def Adam(point=np.zeros(2), lr=0.0009, epoch_count=100, eps=10e-8, beta1=0.9, beta2=0.999):
    s = 0
    v = 0
    for epoch in range(1, epoch_count):
        j = np.random.randint(0, n)
        gr = gradient(point, j)
        v = beta1 * v + (1 - beta1) * gr
        s = beta2 * s + (1 - beta2) * (gr * gr)
        vv = v / (1 - beta1 ** (epoch + 1))
        ss = s / (1 - beta2 ** (epoch + 1))
        point = point - lr * vv / (np.sqrt(ss + eps))
    return point
