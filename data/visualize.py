import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# 读取 CSV 文件
X_bias = np.load('data/example1_D1_M10000/X_bias.npy')
f_star = np.load('data/example1_D1_M10000/f_star.npy')
X = X_bias[:, 0]

# 读取模型
plt.figure(figsize=(10, 6))
plt.scatter(X, f_star, label='f_star vs X')
plt.xlabel('X')
plt.ylabel('f_star')
plt.title('Plot of f_star vs X')
plt.legend()
plt.grid(True)
plt.show()