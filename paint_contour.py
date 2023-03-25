import matplotlib.pyplot as plt
import numpy as np


def paint_contour(points, f):
    lower_bound = -10
    upper_bound = 10
    accuracy = 200
    t1 = np.linspace(lower_bound, upper_bound, accuracy)
    t2 = np.linspace(lower_bound, upper_bound, accuracy)
    A, B = np.meshgrid(t1, t2)
    # F = np.zeros(A.shape)
    # for i in range(len(A)):
    #     for j in range(len(B)):
    #         F[i][j] = f([A[i][j], B[i][j]])
    plt.contour(A, B, f([A, B]), levels=sorted(list(set([f(p) for p in points]))))
    plt.plot(points[:, 0], points[:, 1], '-r8')
    plt.show()
