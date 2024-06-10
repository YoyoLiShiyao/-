#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

# 音板几何参数
length = 0.5  # 长度 (m)
width = 0.5  # 宽度 (m)
thickness = 0.01  # 厚度 (m)

# 网格参数
Nx = 50  # 网格点数（x方向）
Ny = 50  # 网格点数（y方向）

# 材料参数
materials = {
    "Spruce": {"rho": 450, "E": 11e9, "nu": 0.35},
    "Aluminum": {"rho": 2700, "E": 70e9, "nu": 0.33},
    "CarbonFiber": {"rho": 1600, "E": 110e9, "nu": 0.27},
    "NewMaterial": {"rho": 1200, "E": 50e9, "nu": 0.30},
}

# 计算弯曲刚度
def calc_D(E, h, nu):
    return E * h**3 / (12 * (1 - nu**2))

# 构建离散系统矩阵
def build_matrices(material, length, width, thickness, Nx, Ny):
    E = material["E"]
    rho = material["rho"]
    nu = material["nu"]
    D = calc_D(E, thickness, nu)
    
    dx = length / (Nx - 1)
    dy = width / (Ny - 1)
    
    K = np.zeros((Nx * Ny, Nx * Ny))
    M = np.zeros((Nx * Ny, Nx * Ny))
    
    for i in range(Nx):
        for j in range(Ny):
            index = i * Ny + j
            if i > 0:
                K[index, index] += D / dx**2
                K[index, index - Ny] -= D / dx**2
            if i < Nx - 1:
                K[index, index] += D / dx**2
                K[index, index + Ny] -= D / dx**2
            if j > 0:
                K[index, index] += D / dy**2
                K[index, index - 1] -= D / dy**2
            if j < Ny - 1:
                K[index, index] += D / dy**2
                K[index, index + 1] -= D / dy**2
            
            M[index, index] = rho * thickness * dx * dy
    
    return K, M

# 计算振动模态频率和振型
def calculate_modes(material, length, width, thickness, Nx, Ny, num_modes=5):
    K, M = build_matrices(material, length, width, thickness, Nx, Ny)
    eigenvalues, eigenvectors = eigh(K, M)
    
    frequencies = np.sqrt(eigenvalues) / (2 * np.pi)
    modes = [(frequencies[i], eigenvectors[:, i].reshape((Nx, Ny))) for i in range(num_modes)]
    
    return modes

# 计算并对比不同材料的振动模态
results = {}
for name, props in materials.items():
    modes = calculate_modes(props, length, width, thickness, Nx, Ny)
    results[name] = modes

# 输出结果
for material, modes in results.items():
    print(f"Material: {material}")
    for i, (frequency, mode_shape) in enumerate(modes):
        print(f"  Mode {i+1}: Frequency = {frequency:.2f} Hz")
        print(np.array2string(mode_shape, formatter={'float_kind':lambda x: "%.2f" % x}))

# 可视化
fig, axes = plt.subplots(len(materials), 5, figsize=(20, 15))

for i, (material, modes) in enumerate(results.items()):
    for j, (frequency, mode_shape) in enumerate(modes):
        ax = axes[i, j]
        cax = ax.imshow(mode_shape, cmap='coolwarm')
        ax.set_title(f"{material}\nMode {j+1}\nFreq: {frequency:.2f} Hz")
        fig.colorbar(cax, ax=ax)
        
plt.tight_layout()
plt.show()


# In[ ]:




