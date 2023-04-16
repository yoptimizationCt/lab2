import matplotlib.pyplot as plt
import numpy as np

from paint_contour import paint_contour
from create_function import create_function_XY
n = 1000
X = np.random.rand(n) * 10
Y = 10 * X + np.random.rand(n) * 1


def gradient(point, j):
    return np.array(
        [2 * X[j] * (point[0] * X[j] + point[1] - Y[j]),
         2 * (point[0] * X[j] + point[1] - Y[j])])


def SGD(point=np.zeros(2), lr=0.0009, epoch_count=100):
    points = np.zeros((epoch_count + 1, 2))
    for epoch in range(1, epoch_count + 1):
        j = np.random.randint(0, n)
        gr = gradient(point, j)
        points[epoch] = points[epoch - 1] - lr * gr
    return points


def SGD_with_momentum(point=np.zeros(2), lr=0.0009, v=np.zeros(2), gamma=0.5, epoch_count=100):
    points = np.zeros((epoch_count + 1, 2))
    for epoch in range(1, epoch_count + 1):
        j = np.random.randint(0, n)
        v = gamma * v + (1 - gamma) * gradient(point, j)
        points[epoch] = points[epoch - 1] - lr * v
    return points


def Nesterov(point=np.zeros(2), lr=0.0009, v=np.zeros(2), gamma=0.5, epoch_count=100):
    points = np.zeros((epoch_count + 1, 2))
    for epoch in range(1, epoch_count + 1):
        j = np.random.randint(0, n)
        v = gamma * v + (1 - gamma) * gradient(point - lr * gamma * v, j)
        points[epoch] = points[epoch - 1] - lr * v
    return points


def AdaGrad(point=np.zeros(2), lr=0.1, epoch_count=100, eps=10e-8):
    state_sum = 0
    points = np.zeros((epoch_count + 1, 2))
    for epoch in range(1, epoch_count + 1):
        j = np.random.randint(0, n)
        gr = gradient(point, j)
        state_sum += gr * gr
        points[epoch] = points[epoch - 1] - lr * gr / (np.sqrt(state_sum) + eps)
    return points


def RMS_prop(point=np.zeros(2), lr=0.1, epoch_count=100, alpha=0.5, eps=10e-8):
    points = np.zeros((epoch_count + 1, 2))
    s = 0
    for epoch in range(1, epoch_count + 1):
        j = np.random.randint(0, n)
        gr = gradient(point, j)
        s = alpha * s + (1 - alpha) * (gr * gr)
        points[epoch] = points[epoch - 1] - lr * gr / (np.sqrt(s + eps))
    return points


def Adam(point=np.zeros(2), lr=0.1, epoch_count=100, eps=10e-8, beta1=0.9, beta2=0.999):
    points = np.zeros((epoch_count + 1, 2))
    s = 0
    v = 0
    for epoch in range(1, epoch_count + 1):
        j = np.random.randint(0, n)
        gr = gradient(point, j)
        v = beta1 * v + (1 - beta1) * gr
        s = beta2 * s + (1 - beta2) * (gr * gr)
        vv = v / (1 - beta1 ** (epoch + 1))
        ss = s / (1 - beta2 ** (epoch + 1))
        points[epoch] = points[epoch - 1] - lr * vv / (np.sqrt(ss + eps))
    return points


epoch_count = 1000
points_SGD = SGD(epoch_count=epoch_count)
points_SGD_with_momentum = SGD_with_momentum(epoch_count=epoch_count)
points_Nesterov = Nesterov(epoch_count=epoch_count)
points_AdaGrad = AdaGrad(epoch_count=epoch_count)
points_RMS_prop = RMS_prop(epoch_count=epoch_count)
points_Adam = Adam(epoch_count=epoch_count)

print(points_SGD[epoch_count])
print(points_SGD_with_momentum[epoch_count])
print(points_Nesterov[epoch_count])
print(points_AdaGrad[epoch_count])
print(points_RMS_prop[epoch_count])
print(points_Adam[epoch_count])

paint_contour(0, 1000, 0, 1000, 1000, points_SGD, create_function_XY(X, Y))
plt.savefig("SGD.png")
