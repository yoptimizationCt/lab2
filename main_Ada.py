import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (10, 10)
n = 100
X = np.random.rand(n) * 10
Y = 5 * X + np.random.rand(n) * 20
fig, ax = plt.subplots()
ax.scatter(X, Y)

point = np.zeros(2)
eps = 10e-6
lr = 1
v = np.zeros(2)


def gradient(point, j, X):
    return np.array(
        [2 * X[j] * (point[0] * X[j] + point[1] - Y[j]),
         2 * (point[0] * X[j] + point[1] - Y[j])])


state_sum = 0
for epoch in range(1, 100):
    j = np.random.randint(0, n)
    gr = gradient(point, j, X)
    state_sum += gr*gr
    point = point - lr * gr / (np.sqrt(state_sum) + eps)
    # v = gamma * v_prev - lr * grad
    # point = point
print(point)
ax.plot([0, 10], [point[1], 10 * point[0] + point[1]], color='red', linewidth=5)
plt.show()
