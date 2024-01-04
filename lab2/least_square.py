import numpy as np
from scipy.optimize import curve_fit

# 定义方程
def equation(x, a, b, c):
    D,S = x
    return a * D**b * S**c

# 测量数据
D = np.array([0.302, 0.604, 0.906, 0.302, 0.604,0.902, 0.302, 0.604, 0.906])
S = np.array([0.001, 0.001, 0.001, 0.01, 0.01, 0.01, 0.05, 0.05, 0.05])
concat_param = np.vstack((D,S))
Q_observed = np.array([0.0385, 0.2283, 0.6655, 0.1293, 0.7948, 2.3100, 0.3053, 1.8975, 5.5])

# 初始参数估计
initial_guess = (1.0, 1.0, 1.0)  # 适当地初始化a、b、c的值

# 使用最小二乘法拟合
params, covariance = curve_fit(equation, (D,S), Q_observed, p0=initial_guess)

# 拟合结果
a_fit, b_fit, c_fit = params
#画出拟合结果
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制原始数据点
ax.scatter(D, S, Q_observed, color='red', label='Observed Data')

# 生成网格数据来绘制拟合的曲面
D_grid, S_grid = np.meshgrid(np.linspace(D.min(), D.max(), 3000), 
                             np.linspace(S.min(), S.max(), 3000))
Q_grid = equation((D_grid, S_grid), a_fit, b_fit, c_fit)

# 绘制拟合曲面
ax.plot_surface(D_grid, S_grid, Q_grid, color='blue', alpha=0.5)

ax.set_xlabel('D')
ax.set_ylabel('S')
ax.set_zlabel('Q')
ax.legend()
plt.title('3D Scatter and Fitted Surface')
plt.show()
# for i in range(len(D)):
#     print(f"Q_observed: {Q_observed[i]}, Q_fit: {equation((D[i],S[i]), a_fit, b_fit, c_fit)}")
# print(f"a: {a_fit}, b: {b_fit}, c: {c_fit}")