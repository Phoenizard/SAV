import tensorflow as tf
from tensorflow.keras.optimizers import Optimizer

class SimpleOptimizer(Optimizer):
    def __init__(self, learning_rate=0.01, name="SimpleOptimizer", **kwargs):
        # 正确调用父类构造函数
        super().__init__(name=name, learning_rate=learning_rate, **kwargs)
        self.learning_rate = learning_rate

    def _create_slots(self, var_list):
        # 此优化器不需要额外的槽
        pass

    def _resource_apply_dense(self, grad, var, apply_state=None):
        # 应用梯度更新
        lr = self._get_hyper("learning_rate")
        var.assign_sub(lr * grad)

# 使用您的优化器
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(20,)),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer=SimpleOptimizer(learning_rate=0.01), loss='mse')