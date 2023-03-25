import numpy
import numpy as np
import matplotlib.pyplot as plt


plt.rcParams["figure.figsize"] = (10, 10)

# x_0, y_0 = 1.25, 6.9
# x = np.random.rand(10) * 10
# y = np.random.rand(10) * 10
# t1 = np.linspace(x_0, x_0, 100)
# t2 = np.linspace(y_0, y_0, 100)
# X, Y = np.meshgrid(t1, t2)
# ax = plt.figure().add_subplot(projection='3d')
# ax.plot_surface(X, Y, f([X, Y]))
# ax.set_zlim([None, 6])
# fig, ax = plt.subplots()
# ax.set_xlim(-10, 10)
# ax.set_ylim(-10, 10)
# ax.plot([0, 1], [10, 13])
# ax.scatter(points)

# plt.savefig("sample.png")
# plt.show()
# fig.savefig("sample.png")
n = np.random.randint(10, 100)
X = np.random.rand(n) * 10
Y = np.random.rand(n) * 10

# print(numpy.gradient([[], []]))
fig, ax = plt.subplots()
ax.scatter(X, Y)

# (X[i] * (a + h) + b + h - Y[i]) - (X[i] * a + b - Y[i])

point = [0, 0]
h = 10e-6
lr = 0.09
for epoch in range(1, 100):
    j = np.random.randint(0, n)
    # grad = np.gradient([[X[j] * (point[0] + h * i + point[1] - Y[j] for i in range(0, 2)] for ])
    # a = np.array([X[j] * point[0] + point[1] - Y[j], X[j] * point[0] + point[1] + h - Y[j]])
    # a_h = np.array([X[j] * (point[0] + h) + point[1] - Y[j], X[j] * (point[0] + h) + point[1] + h - Y[j]])
    # grad = np.gradient([a, a_h], h)
    # print(grad)
    # point = point - 10e-7 * grad
    grad = np.array([2 * X[j] * (point[0] * X[j] + point[1] - Y[j]), 2 * (point[0] * X[j] + point[1] - Y[j])])
    point = point - lr * grad

print(point)
ax.plot([0, 10], [point[1], 10 * point[0] + point[1]])
# plt.contour(X, Y, f([X, Y]), levels=sorted(list(set([f(p) for p in points]))))
plt.show()

