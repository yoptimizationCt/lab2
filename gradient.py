import numpy as np

# Определение функции
def f(x, y):
    return x**2 + 2*y**2

# Определение точки, в которой мы хотим вычислить градиент
x0, y0 = 1, 2

# Определение шага для вычисления градиента
h = 1e-6

# Определение интервалов значений для каждого измерения
x_values = np.array([x0 - h, x0, x0 + h])
y_values = np.array([y0 - h, y0, y0 + h])

# Вычисление градиента
a = [[f(x, y) for y in y_values] for x in x_values]
dx, dy = np.gradient(a, h)

# Создание массива из производных
gradient = np.array([dx[1], dy[1]])

# Вывод результата
print("Градиент в точке ({}, {}): {}".format(x0, y0, gradient))