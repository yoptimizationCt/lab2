import numpy as np
import matplotlib.pyplot as plt
from paint_contour import paint_contour
from create_function import create_function

plt.rcParams["figure.figsize"] = (10, 10)


# def random_linear_dependence(a, b, x_scale, points_number, variation):
#     X = np.random.rand(points_number) * x_scale
#     Y = a * X + b + np.random.rand(points_number) * variation
#     return X, Y


# Создаёт полиномиальную зависимость заданной степени с заданным разбросом
def random_polynomial(degree, x_scale, points_number, variation):
    X = (np.random.rand(points_number) - np.random.rand(points_number)) * x_scale
    Y = np.random.rand(points_number) * variation
    for n in range(degree + 1):
        Y += np.random.rand(1) * X ** n
    return X, Y


# linear regression case (2 dimensions)
# def sum_gradient(X, Y, point, summand_numbers):
#     gradient = np.zeros(2)
#     for i in summand_numbers:
#         gradient += np.array([2 * X[i] * (point[0] * X[i] + point[1] - Y[i]), 2 * (point[0] * X[i] + point[1] - Y[i])])
#     return gradient


# Считает значение заданного слагаемого в заданной точке
def calc_summand(x, y, point):
    value = -y
    for i in range(len(point)):
        value += point[i] * x ** i
    return value * value


# Считает значение полинома в заданной точке
def calc_polynomial(x, coefs):
    y = 0
    for n in range(len(coefs)):
        y += coefs[n] * x ** n
    return y


# Считает градиент заданного слагаемого в заданной точке
def get_gradient(x, y, point, h=10e-6):
    gradient = np.zeros(len(point))
    for i in range(len(point)):
        delta = point.copy()
        delta[i] += h
        gradient[i] = (calc_summand(x, y, delta) - calc_summand(x, y, point)) / h
    return gradient


# polynomial regression case
# Считает градиент суммы заданных слагаемых (батча)
def sum_gradient(X, Y, point, summand_numbers):
    gradient = np.zeros(len(point))
    for i in summand_numbers:
        gradient += get_gradient(X[i], Y[i], point)
    return gradient


def gradient_descent(X, Y, start_point, learning_rate, epochs, batch_size):
    points = np.zeros((epochs, len(start_point)))
    points[0] = start_point
    lr = learning_rate
    for epoch in range(1, epochs):
        summand_numbers = [(epoch * batch_size + j) % len(X) for j in range(batch_size)]
        points[epoch] = points[epoch - 1] - lr * sum_gradient(X, Y, points[epoch - 1], summand_numbers)
        # Exponential decay
        # lr *= 0.95
        # Step decay
        if epoch % 100 == 0:
            lr *= 0.5
    return points


# Результат ОЧЕНЬ сильно зависит от шага, кол-ва точек, разброса точек, размера батча
degree = 2
n = 10
x_scale = 1
start_point = np.zeros(degree + 1)
learning_rate = 0.09
epochs = 1000
batch_size = 5

# X, Y = random_linear_dependence(a=3, b=0, x_scale=x_scale, points_number=n, variation=20)
# X, Y = random_polynomial(degree, x_scale, n, 0.1)

# descent_points = np.zeros((epochs, degree + 1))

X, Y = random_polynomial(degree, x_scale, n, 0.3)
descent_points = gradient_descent(X, Y, start_point, learning_rate, epochs, batch_size)

# all_points - нужен двумерный массив точек в create_function, а X, Y - одномерные массивы :(
# all_points = np.zeros((n, 2))
# for i in range(n):
#     all_points[i][0] = X[i]
#     all_points[i][1] = Y[i]
# paint_contour(-5, 15, 0, 20, 200, descent_points, create_function(all_points))
# plt.savefig("descent_default.png")
# plt.cla()

plt.scatter(X, Y)

min_point = descent_points[-1]
X_values = np.linspace(-x_scale, x_scale, 200)
# plt.plot([0, x_scale], [min_point[1], x_scale * min_point[0] + min_point[1]], color='red', linewidth=5)
plt.plot(X_values, calc_polynomial(X_values, min_point), color='red')
plt.savefig("regression_default.png")
plt.cla()
