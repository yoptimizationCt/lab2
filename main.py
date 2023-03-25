import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (10, 10)

n = 1000
X = np.random.rand(n) * 10
Y = 5 * X + np.random.rand(n) * 20

point = [0, 0]
h = 10e-6
lr = 0.0009
for epoch in range(1, 100):
    j = np.random.randint(0, n)
    grad = np.array([2 * X[j] * (point[0] * X[j] + point[1] - Y[j]), 2 * (point[0] * X[j] + point[1] - Y[j])])
    point = point - lr * grad

print(point)

fig, ax = plt.subplots()
ax.scatter(X, Y)

ax.plot([0, 10], [point[1], 10 * point[0] + point[1]], color='red', linewidth=5)
plt.show()
