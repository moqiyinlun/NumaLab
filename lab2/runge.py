import numpy as np
import matplotlib.pyplot as plt

# 更新代码以使用一组更好看的曲线颜色

# 定义原始函数 f(x)
def f(x):
    return 1 / (1 + 25 * x**2)

# 定义Lagrange插值函数
def lagrange_interpolation(x, xi, yi):
    n = len(xi)
    result = 0.0
    for j in range(n):
        term = 1.0
        for i in range(n):
            if i != j:
                term *= (x - xi[i]) / (xi[j] - xi[i])
        result += yi[j] * term
    return result

# 在[-1, 1]上生成等间距的插值点
x_interpolation = np.linspace(-1, 1, 1000)

# 不同阶数的插值
orders = [2, 5, 10, 15]
plt.figure(figsize=(10, 6))

# 设置一组好看的曲线颜色
colors = ['blue', 'green', 'red', 'purple']

for i, order in enumerate(orders):
    # 生成插值节点
    xi = np.linspace(-1, 1, order + 1)
    yi = [f(x) for x in xi]

    # 计算插值结果
    y_interpolation = [lagrange_interpolation(x, xi, yi) for x in x_interpolation]

    # 绘制插值结果
    plt.plot(x_interpolation, y_interpolation, label=f'Order {order}', color=colors[i])

# 绘制原始函数
x_original = np.linspace(-1, 1, 1000)
y_original = [f(x) for x in x_original]
plt.plot(x_original, y_original, label='Original Function', linestyle='--', color='black')

plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Lagrange Interpolation')
plt.legend()
plt.grid()
plt.savefig("runge_origin.png")