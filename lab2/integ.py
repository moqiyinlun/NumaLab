import numpy as np
from scipy.integrate import quad
from scipy.special import roots_legendre
import matplotlib.pyplot as plt
# 定义Simpson求积公式
def simpson_integral(f, a, b, n=3):
    x = np.linspace(a, b, n)
    h = (b - a) / (n - 1)
    integral = f(a) + f(b)
    integral += 2 * np.sum(f(x[2:n-1:2]))
    integral += 4 * np.sum(f(x[1:n:2]))
    return integral * h / 3

# 定义3点Gauss-Legendre求积公式
def gauss_legendre_3pt(f, a, b):
    x, w = roots_legendre(3)
    print(x,w)
    
    x = 0.5 * (x + 1) * (b - a) + a  # 映射到积分区间
    return np.sum(w * f(x)) * 0.5 * (b - a)

# 定义给定的积分方程中的积分部分函数
def integral_func1(t, x, y_func):
    return (1 / (1 + t) - x) * y_func(t)

def integral_func2(t, x, y_func):
    return (t - x) * y_func(t)

# 精确解函数
def exact_solution1(x):
    return 1 / (1 + x) ** 2

def exact_solution2(x):
    return np.exp(2 * x)

# 使用数值方法求解积分方程的近似解
def approximate_solution(func, method, y_func, x_range, extra_term):
    approx_sol = []
    for x in x_range:
        integral = method(lambda t: func(t, x, y_func), 0, 1)
        approx_sol.append(integral + extra_term(x))
    return approx_sol

# 定义额外的项
extra_term1 = lambda x: (4 * x**3 + 5 * x**2 - 2 * x + 5) / (8 * (x + 1)**2)
extra_term2 = lambda x: np.exp(2 * x) + ((np.exp(2) - 1) / 2) * x - (np.exp(2) + 1) / 4

# 设置x的范围
x_range = np.linspace(0, 10, 10000)

# 计算近似解
approx_sol1_simpson = approximate_solution(integral_func1, simpson_integral, exact_solution1, x_range, extra_term1)
approx_sol1_gauss = approximate_solution(integral_func1, gauss_legendre_3pt, exact_solution1, x_range, extra_term1)

approx_sol2_simpson = approximate_solution(integral_func2, simpson_integral, exact_solution2, x_range, extra_term2)
approx_sol2_gauss = approximate_solution(integral_func2, gauss_legendre_3pt, exact_solution2, x_range, extra_term2)

# 返回结果供后续展示
# print(approx_sol1_simpson[:5])#, approx_sol1_gauss[:5], approx_sol2_simpson[:5], approx_sol2_gauss[:5]  # 展示部分结果
# 定义用于可视化的x范围

# 计算精确解的函数值
exact_sol1 = [exact_solution1(x) for x in x_range]
exact_sol2 = [exact_solution2(x) for x in x_range]

# 这里假设approx_sol1_simpson, approx_sol1_gauss, approx_sol2_simpson, approx_sol2_gauss是你之前计算得到的近似解列表
# 如果没有这些数据，你需要根据前面的方法重新计算这些值

# 绘制第一个积分方程的近似解和精确解
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(x_range, exact_sol1, label='Exact Solution', color='black')
plt.plot(x_range, approx_sol1_simpson, label='Simpson Approximation', linestyle='--')
plt.plot(x_range, approx_sol1_gauss, label='Gauss-Legendre Approximation', linestyle='-.')
plt.title('Approximations vs Exact Solution for Equation 1')
plt.xlabel('x')
plt.ylabel('y(x)')
plt.legend()

# 绘制第二个积分方程的近似解和精确解
plt.subplot(1, 2, 2)
plt.plot(x_range, exact_sol2, label='Exact Solution', color='black')
plt.plot(x_range, approx_sol2_simpson, label='Simpson Approximation', linestyle='--')
plt.plot(x_range, approx_sol2_gauss, label='Gauss-Legendre Approximation', linestyle='--')
plt.title('Approximations vs Exact Solution for Equation 2')
plt.xlabel('x')
plt.ylabel('y(x)')
plt.legend()

plt.tight_layout()
plt.show()