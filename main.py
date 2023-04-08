import numpy as np
import matplotlib.pyplot as plt
from paint_contour import paint_contour
from create_function import create_function

plt.rcParams["figure.figsize"] = (10, 10)


def random_linear_dependence(a, b, x_scale, points_number, variation):
    X = np.random.rand(points_number) * x_scale
    Y = a * X + b + np.random.rand(points_number) * variation
    return X, Y


def sum_gradient(X, Y, point, summand_numbers):
    gradient = np.zeros(2)
    for i in summand_numbers:
        gradient += np.array([2 * X[i] * (point[0] * X[i] + point[1] - Y[i]), 2 * (point[0] * X[i] + point[1] - Y[i])])
    return gradient


def gradient_descent(X, Y, start_point, learning_rate, epochs, batch_size):
    points = np.zeros((epochs, 2))
    points[0] = start_point
    for epoch in range(1, epochs):
        # summand_numbers = np.random.randint(0, len(X), batch_size)
        summand_numbers = [(epoch * batch_size + j) % len(X) for j in range(batch_size)]
        points[epoch] = points[epoch - 1] - learning_rate * sum_gradient(X, Y, points[epoch - 1], summand_numbers)
    return points


# h = 10e-6
n = 20
x_scale = 10
start_point = np.random.rand(2) * 10
# start_point = [10, 10]
learning_rate = 0.0009
epochs = 1000
batch_size = 10

X, Y = random_linear_dependence(a=3, b=0, x_scale=x_scale, points_number=n, variation=20)

# descent_points = np.zeros((epochs, 2))
# for batch_size in range(1, n + 1):
descent_points = gradient_descent(X, Y, start_point, learning_rate, epochs, batch_size)

# all_points - нужен двумерный массив точек в create_function, а X, Y - одномерные массивы :(
all_points = np.zeros((n, 2))
for i in range(n):
    all_points[i][0] = X[i]
    all_points[i][1] = Y[i]
paint_contour(-5, 15, 0, 20, 200, descent_points, create_function(all_points))
# fig, ax = plt.subplots()
plt.title("Batch size = " + str(batch_size))
plt.savefig("batch/descent_batch_" + str(batch_size) + ".png")
plt.cla()

plt.scatter(X, Y)

min_point = descent_points[-1]
plt.plot([0, x_scale], [min_point[1], x_scale * min_point[0] + min_point[1]], color='red', linewidth=5)
plt.savefig("batch/regression.png")
plt.cla()