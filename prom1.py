import numpy as np
from scipy.optimize import fsolve
from math import cos, sin, exp, pi

def F(x):
    return [
        3 * x[0] - cos(x[1] * x[2]) - 0.5,
        x[0]**2 - 81 * (x[1] + 0.1) + sin(x[2]) + 1.06,
        exp(-x[0] * x[1]) + 20 * x[2] + (10 * pi - 3) / 3
    ]
def jacobian(x):
    return [
        [3, x[2] * sin(x[1] * x[2]), x[1] * sin(x[1] * x[2])],
        [2 * x[0], -81, cos(x[2])],
        [-x[1] * exp(-x[0] * x[1]), -x[0] * exp(-x[0] * x[1]), 20]
    ]

def jacobi_method(x0, epsilon=1e-9, max_iterations=10000000):
    x = np.array(x0)
    for _ in range(max_iterations):
        x_old = np.array(x)
        x[0] = (cos(x_old[1] * x_old[2]) + 0.5) / 3
        x[1] = (x_old[0]**2 + sin(x_old[2]) + 1.06)/ 81 - 0.1
        x[2] = -((10 * pi - 3) / 3 + exp(-x_old[0] * x_old[1])) / 20
        if np.linalg.norm(x - x_old, ord=np.inf) < epsilon:
            return x
    return x
def gauss_seidel_method(x0, epsilon=1e-9, max_iterations=10000000):
    x = np.array(x0)
    for _ in range(max_iterations):
        x_old = np.array(x)
        x[0] = (cos(x[1] * x[2]) + 0.5) / 3
        x[1] =  (x[0]**2 + sin(x[2]) + 1.06) / 81 - 0.1
        x[2] = -((10 * pi - 3) / 3 + exp(-x[0] * x[1])) / 20
        if np.linalg.norm(x - x_old, ord=np.inf) < epsilon:
            return x
    return x
def newton_method(F, jacobian, x0, epsilon=1e-5, max_iterations=1000):
    x = np.array(x0)
    for _ in range(max_iterations):
        J = jacobian(x)
        Fx = F(x)
        x = x - np.linalg.inv(J) @ np.array(Fx)
        if np.linalg.norm(np.linalg.inv(J) @ np.array(Fx), ord=np.inf) < epsilon:
            return x
    return x
x0 = [0.1, 0.1, -0.1]
jacobi_solution = jacobi_method(x0)
print("Jacobi 不动点法解:", jacobi_solution)
print(F(jacobi_solution))

gauss_seidel_solution = gauss_seidel_method(x0)
print("Gauss-Seidel 不动点法解:", gauss_seidel_solution)
print(F(gauss_seidel_solution))
newton_solution = newton_method(F, jacobian, [0.1, 0.1, -0.1])
print("Newton 迭代法解:", newton_solution)
print(F(newton_solution))