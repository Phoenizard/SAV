import numpy as np
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
np.random.seed(100)
D = 1
M = 200
file_name = 'example1_D1_M200'
dir_path = os.path.join('data', file_name)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

def data_gripper(D, M):
    p = np.random.randn(D)
    q = np.random.randn(D)
    X = np.random.rand(M, D)
    f_star = np.sin(X.dot(p)) + np.cos(X.dot(q))
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
    X_bias = np.hstack([X_normalized, np.ones((M, 1))])
    return X_bias, f_star, p, q

X_bias, f_star, p, q = data_gripper(D, M)

content = f"p: {p.tolist()}\nq: {q.tolist()}"
# 将字符串写入 txt 文件
with open('data/'+ file_name+'/p_q.txt', 'w') as file:
    file.write(content)

# 假设 X_bias 和 f_star 是你的数据数组
np.save('data/'+ file_name + '/X_bias.npy', X_bias)
np.save('data/' + file_name + '/f_star.npy', f_star)