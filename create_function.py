def create_function_XY(X, Y):
    # a^2 * \sum x_i^2 + b^2 * n + \sum y_i^2 + 2ab * \sum x_i - 2a*\sum x_i*y_i - 2b \sum y_i
    n = len(X)
    summ_x_sqr = 0
    summ_y_sqr = 0
    summ_x = 0
    summ_y = 0
    summ_x_y = 0
    for i in range(n):
        x = X[i]
        y = Y[i]
        summ_x += x
        summ_y += y
        summ_x_y += x * y
        summ_x_sqr += x * x
        summ_y_sqr += y * y
    return lambda arg: arg[0] ** 2 * summ_x_sqr + arg[1] ** 2 * n + summ_y_sqr + 2 * summ_x * arg[0] * arg[1] - 2 * arg[
        0] * summ_x_y - 2 * arg[1] * summ_y


def create_function(points):
    # a^2 * \sum x_i^2 + b^2 * n + \sum y_i^2 + 2ab * \sum x_i - 2a*\sum x_i*y_i - 2b \sum y_i
    n = len(points)
    summ_x_sqr = 0
    summ_y_sqr = 0
    summ_x = 0
    summ_y = 0
    summ_x_y = 0
    for [x, y] in points:
        summ_x += x
        summ_y += y
        summ_x_y += x * y
        summ_x_sqr += x * x
        summ_y_sqr += y * y
    return lambda arg: arg[0] ** 2 * summ_x_sqr + arg[1] ** 2 * n + summ_y_sqr + 2 * summ_x * arg[0] * arg[1] - 2 * arg[
        0] * summ_x_y - 2 * arg[1] * summ_y
