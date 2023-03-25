import numpy as np
import matplotlib.pyplot as plt
from paint_contour import paint_contour
from create_function import create_function

plt.rcParams["figure.figsize"] = (10, 10)

n = 10
X = np.random.rand(n) * 10
Y = 10 - 5*X * np.random.rand(n) * 2


def summand_gradient(point, ind):
    return np.array([2 * X[ind] * (point[0] * X[ind] + point[1] - Y[ind]), 2 * (point[0] * X[ind] + point[1] - Y[ind])])


def sum_gradient(point, summand_numbers):
    gradient = np.zeros(2)
    for ind in summand_numbers:
        gradient += summand_gradient(point, ind)
    return gradient


cur_x = [0, 0]
h = 10e-6
lr = 0.0009
epochs = 1000
points = np.zeros((epochs, 2))
points[0] = cur_x
for epoch in range(1, epochs):
    j = np.random.randint(0, n)
    grad = np.array([2 * X[j] * (cur_x[0] * X[j] + cur_x[1] - Y[j]), 2 * (cur_x[0] * X[j] + cur_x[1] - Y[j])])
    cur_x = cur_x - lr * grad
    points[epoch] = cur_x

print(cur_x)

fig, ax = plt.subplots()
ax.scatter(X, Y)

ax.plot([0, 10], [cur_x[1], 10 * cur_x[0] + cur_x[1]], color='red', linewidth=5)
plt.show()
# all_points - нужен двумерный массив точек в create_function, а X, Y - одномерные массивы :(
all_points = np.zeros((n, 2))
for i in range(n):
    all_points[i][0] = X[i]
    all_points[i][1] = Y[i]
paint_contour(points, create_function(all_points))
