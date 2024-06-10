#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

# 设定网格范围和细节
x = np.linspace(-10, 10, 400)
y = np.linspace(-10, 10, 400)
x, y = np.meshgrid(x, y)

# 模态函数定义
def mode1(x, y, A, omega):
    return A * np.exp(-0.1*(x**2 + y**2)) * np.cos(omega * x)

def mode2(x, y, A, omega):
    return A * (np.sin(omega * x) * np.cos(omega * y))

def mode3(x, y, A, omega):
    return A * np.sin(omega * x) * np.cos(omega * y) + A * y * np.exp(-x**2)

def mode4(x, y, A, omega):
    return A * np.cos(omega * x) * np.exp(-0.1 * y**2)

def mode5(x, y, A, omega):
    return A * (np.sin(omega * x) * np.cos(omega * y)) + A * (np.sin(omega * y) * np.cos(omega * x))

# 计算每个模态的值
z1 = mode1(x, y, 1, 1)
z2 = mode2(x, y, 0.8, 2)
z3 = mode3(x, y, 0.6, 3)
z4 = mode4(x, y, 0.4, 4)
z5 = mode5(x, y, 0.2, 5)

# 绘图函数
def plot_mode(x, y, z, title):
    plt.figure(figsize=(8, 6))
    cp = plt.contourf(x, y, z, levels=20, cmap='viridis')
    plt.colorbar(cp)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('scaled')

# 绘制每个模态
plot_mode(x, y, z1, 'Mode 1: 82 Hz')
plot_mode(x, y, z2, 'Mode 2: 158 Hz')
plot_mode(x, y, z3, 'Mode 3: 218 Hz')
plot_mode(x, y, z4, 'Mode 4: 231 Hz')
plot_mode(x, y, z5, 'Mode 5: 331 Hz')

plt.show()


# In[ ]:




