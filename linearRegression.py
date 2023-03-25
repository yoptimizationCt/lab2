import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # Инициализация начальных весов и смещения
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Стохастический градиентный спуск
        for i in range(self.n_iters):
            # Выбор случайного примера
            rand_idx = np.random.randint(n_samples)
            X_i = X[rand_idx, :]
            y_i = y[rand_idx]

            # Рассчет ошибки и градиента функции потерь
            y_pred = np.dot(X_i, self.weights) + self.bias
            error = y_i - y_pred
            dw = -2 * np.dot(X_i.T, error)
            db = -2 * error

            # Обновление весов и смещения
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred

# Генерация синтетических данных
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)



# Создание и обучение модели
model = LinearRegression(lr=0.01, n_iters=1000)
model.fit(X, y)

# Предсказание на новых данных
X_new = np.array([[0], [2]])
y_pred = model.predict(X_new)
print(y_pred)

plt.plot(X, y, "b.")
plt.plot(X_new, y_pred, "r-", linewidth=2, label="graph")
plt.xlabel("X")
plt.ylabel("y")
plt.legend(loc="upper left", fontsize=14)
plt.axis([0, 2, 0, 15])
plt.show()
plt.savefig("hehe.png")
