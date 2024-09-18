import torch
import torch.nn as nn
from model.model_torch import Model
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def relative_error(y_true, y_pred):
    """计算相对误差"""
    return torch.mean(torch.norm(y_pred - y_true) / torch.norm(y_true))

m = 50
D = 1

# 创建模型
model = Model(m, D)
model.load_state_dict(torch.load('save/SGD_D1_10000_1_torch.pth'))

X_bias = torch.from_numpy(np.load('data/example1_D1_M10000/X_bias.npy')).float()
f_star = torch.from_numpy(np.load('data/example1_D1_M10000/f_star.npy')).float().reshape(-1, 1)
X = X_bias[:, 0]
pred = model(X_bias)

error = relative_error(f_star, pred)
print(f'Relative error: {error.item()}')

# 转化为 numpy 数组绘图
X = X.numpy()
f_star = f_star.numpy()
pred = pred.detach().numpy()

# 可视化散点图
plt.figure(figsize=(10, 6))
plt.scatter(X, f_star, label='f_star vs X')
plt.scatter(X, pred, label='pred vs X')
plt.xlabel('X')
plt.ylabel('f_star')
plt.title('Plot of f_star vs X')
plt.legend()
plt.grid(True)
plt.show()