import tensorflow as tf
from tensorflow.keras.layers import Conv2D, ReLU

from BatchNormalize import BatchNorm


class ConvBNRelu(tf.keras.layers.Layer):
    def __init__(self, name, filters, kernel_size, strides=1, padding='SAME', bn_input_shape=()):
        super().__init__()
        self.batch_norm = None
        self.conv = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                           padding=padding, name=name)
        
        self.relu = ReLU(name=f'{name}_relu')

    def build(self, bn_input_shape):
        self.batch_norm = BatchNorm(input_shape=bn_input_shape, name=f'{self.name}_batch')

    def call(self, inputs, **kwargs):
        bridged_input = inputs
        bridged_input = self.conv(bridged_input)
        bridged_input = self.batch_norm(bridged_input)
        bridged_input = self.relu(bridged_input)
        return bridged_input
