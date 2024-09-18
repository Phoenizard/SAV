import tensorflow as tf
import numpy as np
import json
import pandas as pd
from model.model_tf import create_model
#=========================CallBack================================
class PrintLossCallback(tf.keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        print(f"Epoch {self.params['epochs']}, Batch {batch}, Loss: {logs['loss']}")
#=========================Set Seed================================
np.random.seed(100)
tf.random.set_seed(100)
#=========================Load Data================================
X_bias = np.load('data/example1_D1_M1000/X_bias.npy')
f_star = np.load('data/example1_D1_M1000/f_star.npy')
M = X_bias.shape[0]
split_index = int(M * 0.8)
X_train, X_test = X_bias[:split_index], X_bias[split_index:]
y_train, y_test = f_star[:split_index], f_star[split_index:]
#=========================Load Model===============================
D = X_bias.shape[1] - 1
m = 1000
model = create_model(m, D)
#=======================Configure Training=========================
batch_size = 32
learning_rate = 1.0
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='mse')
#=======================Train Model=================================
history = model.fit(X_train, y_train, 
                    batch_size=batch_size, 
                    epochs=10000, 
                    validation_data=(X_test, y_test))
#=======================Save Model==================================
is_save = False
if is_save:
    name = 'SGD_D1_32_1e3_tf'
    history_dict = history.history
    model.save("save/" + name + ".keras")
    # 保存为 JSON 文件
    with open(f'save/{name}_history.json', 'w') as f:
        json.dump(history_dict, f)