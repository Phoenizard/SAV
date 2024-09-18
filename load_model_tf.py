import tensorflow as tf
from model.model_tf import create_model
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def relative_error(y_true, y_pred):
    """计算相对误差"""
    return tf.reduce_mean(tf.norm(y_pred - y_true) / tf.norm(y_true))

m = 1000
D = 1

# 创建模型
model = create_model(m, D)
model.load_weights('save/SGD_D1_32_2e1.keras')

X_bias = np.load('data/example1_D1_M1000/X_bias.npy')
f_star = np.load('data/example1_D1_M1000/f_star.npy')
X = X_bias[:, 0]
pred = model.predict(X_bias)

# 计算相对误差
error = relative_error(f_star, pred)
print(f'Relative error: {error.numpy()}')


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