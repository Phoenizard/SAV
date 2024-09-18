import tensorflow as tf

class DivisionLayer(tf.keras.layers.Layer):
    def __init__(self, divisor):
        super(DivisionLayer, self).__init__()
        self.divisor = divisor  # 定义除数

    def call(self, inputs):
        return inputs / self.divisor

def create_model(m, D):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=m, input_dim=D+1, 
                            kernel_initializer=tf.keras.initializers.HeNormal(), 
                            activation=None),
        tf.keras.layers.ReLU(),  # ReLU激活函数
        tf.keras.layers.Dense(units=1, 
                            kernel_initializer=tf.keras.initializers.HeNormal(),
                            activation=None),  # 输出层
        DivisionLayer(divisor=m)
    ])
    return model