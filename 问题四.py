#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 设置字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 已知的模态频率（来自附件）
target_freqs = np.array([82, 158, 218, 231, 331])

# 初始材料参数
initial_params = {
    'rho': 7800,  # 密度
    'E': 2.1e11,  # 杨氏模量
    'nu': 0.3,    # 泊松比
    'h': 0.01     # 厚度
}

# 定义问题参数
Lx = 1.0  # 板的长度
Ly = 1.0  # 板的宽度
nx = 30   # x方向网格数
ny = 30   # y方向网格数

# 生成网格
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)

# 定义网格和刚度矩阵、质量矩阵生成函数
def generate_fem_matrices(nx, ny, E, nu, rho, h, Lx, Ly):
    K = sp.lil_matrix((nx * ny, nx * ny))
    M = sp.lil_matrix((nx * ny, nx * ny))
    D = E * h**3 / (12 * (1 - nu**2))

    for i in range(nx):
        for j in range(ny):
            n = i * ny + j
            if i < nx-1:
                m = (i+1) * ny + j
                K[n, n] += D / (Lx/nx)**3
                K[n, m] -= D / (Lx/nx)**3
                K[m, n] -= D / (Lx/nx)**3
                K[m, m] += D / (Lx/nx)**3
            if j < ny-1:
                m = i * ny + (j+1)
                K[n, n] += D / (Ly/ny)**3
                K[n, m] -= D / (Ly/ny)**3
                K[m, n] -= D / (Ly/ny)**3
                K[m, m] += D / (Ly/ny)**3
            M[n, n] = rho * h * (Lx/nx) * (Ly/ny) / 4

    return K.tocsr(), M.tocsr()

# 目标函数
def objective(params):
    E = params[0]
    rho = params[1]
    h = params[2]
    
    K, M = generate_fem_matrices(nx, ny, E, initial_params['nu'], rho, h, Lx, Ly)
    eigenvalues, _ = spla.eigsh(K, M=M, k=5, which='SM')
    
    # 处理可能的负值和零值特征值
    valid_eigenvalues = eigenvalues[eigenvalues > 1e-8]
    if len(valid_eigenvalues) < 5:
        return np.inf  # 如果有效的特征值不足，则返回一个大的误差
    
    freqs = np.sqrt(valid_eigenvalues)
    
    # 计算目标频率和当前频率的差异
    error = np.sum((target_freqs[:len(freqs)] - freqs)**2)
    return error

# 优化材料参数
result = minimize(objective, [initial_params['E'], initial_params['rho'], initial_params['h']],
                  bounds=[(1e10, 3e11), (1000, 10000), (0.005, 0.02)])

optimized_E = result.x[0]
optimized_rho = result.x[1]
optimized_h = result.x[2]

print("优化后的材料参数：")
print(f"杨氏模量：{optimized_E:.2e} Pa")
print(f"密度：{optimized_rho:.2f} kg/m^3")
print(f"厚度：{optimized_h:.3f} m")

# 重新生成刚度矩阵和质量矩阵，计算最终的模态频率和振型
K, M = generate_fem_matrices(nx, ny, optimized_E, initial_params['nu'], optimized_rho, optimized_h, Lx, Ly)
eigenvalues, eigenvectors = spla.eigsh(K, M=M, k=5, which='SM')
final_freqs = np.sqrt(np.maximum(eigenvalues, 0))  # 确保特征值非负

# 输出最终的模态频率
print("最终的模态频率：")
for i, freq in enumerate(final_freqs):
    print(f"模态 {i+1} 频率: {freq:.2f} Hz")

# 对比频率
comparison = np.vstack((target_freqs, final_freqs)).T

print("已知频率和计算频率对比:")
print(comparison)

# 绘制最终的模态振型
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

for i in range(5):
    mode_shape = eigenvectors[:, i].reshape((nx, ny))
    ax = axs[i//3, i%3]
    c = ax.contourf(X, Y, mode_shape, cmap='viridis')
    ax.set_title(f'模态 {i+1} - 频率: {final_freqs[i]:.2f} Hz')
    fig.colorbar(c, ax=ax)

fig.delaxes(axs[1, 2])
plt.tight_layout()
plt.show()


# In[ ]:




