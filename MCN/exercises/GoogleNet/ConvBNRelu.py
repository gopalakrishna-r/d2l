import tensorflow as tf
from tensorflow.keras.layers import Conv2D, ReLU

from BatchNormalize import BatchNorm


class ConvBNRelu(tf.keras.layers.Layer):
    def __init__(self, name, filters, kernel_size, strides, padding):
        super().__init__()
        self.batch_norm = None
        self.conv = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                           padding=padding, name=name)
        self.batch_norm = BatchNorm(name=f'{self.name}_batch')
        self.relu = ReLU(name=f'{name}_relu')

    def call(self, inputs, **kwargs):
        bridged_input = inputs
        bridged_input = self.conv(bridged_input)
        bridged_input = self.batch_norm(bridged_input, training=tf.keras.backend.learning_phase())
        bridged_input = self.relu(bridged_input)
        return bridged_input
