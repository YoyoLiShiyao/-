#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt

# 定义物理参数
E = 11e9  # 杨氏模量 (Pa)
rho = 450  # 密度 (kg/m^3)
nu = 0.35  # 泊松比
h0 = 0.005  # 基础厚度 (m)
delta_h = 0.001  # 厚度变化量 (m)
alpha = 100  # 厚度变化速率

# 创建网格
n = 30  # 网格大小
x = np.linspace(-0.1, 0.1, n)
y = np.linspace(-0.1, 0.1, n)
X, Y = np.meshgrid(x, y)

# 计算每个点的厚度
H = h0 + delta_h * np.exp(-alpha * (X**2 + Y**2))

# 汇总到全局刚度矩阵和质量矩阵
K = np.zeros((n*n, n*n))
M = np.zeros((n*n, n*n))

element_area = (x[1] - x[0]) * (y[1] - y[0])

for i in range(n-1):
    for j in range(n-1):
        # 为每个元素计算厚度
        h_mean = np.mean([H[i, j], H[i, j+1], H[i+1, j], H[i+1, j+1]])
        # 计算局部刚度矩阵和质量矩阵
        D_local = E * h_mean**3 / (12 * (1 - nu**2))
        rho_local = rho * h_mean
        K_local = D_local * np.array([[4, -2, -1, -1], [-2, 4, -1, -1], [-1, -1, 4, -2], [-1, -1, -2, 4]]) / element_area
        M_local = rho_local * element_area * np.array([[2, 1, 1, 1], [1, 2, 1, 1], [1, 1, 2, 1], [1, 1, 1, 2]]) / 4

        indices = [i*n+j, i*n+(j+1), (i+1)*n+j, (i+1)*n+(j+1)]
        for ii in range(4):
            for jj in range(4):
                K[indices[ii], indices[jj]] += K_local[ii, jj]
                M[indices[ii], indices[jj]] += M_local[ii, jj]

# 求解特征值问题
from scipy.linalg import eigh
eigenvalues, eigenvectors = eigh(K, M)
frequencies = np.sqrt(np.abs(eigenvalues)) / (2 * np.pi)

# 显示第一个振型
mode_number = 0
mode_shape = eigenvectors[:, mode_number].reshape(n, n)
plt.contourf(X, Y, mode_shape, levels=50, cmap='viridis')
plt.colorbar()
plt.title(f'Vibration Mode Shape for Frequency: {frequencies[mode_number]:.2f} Hz')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.show()


# In[ ]:




