from ConvBNRelu import ConvBNRelu
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, MaxPool2D


class Blk2(Layer):
    def __init__(self, name):
        super().__init__()
        self.conv = ConvBNRelu(name=f'{name}_conv', filters=64, kernel_size=1, strides=1, padding='VALID')
        self.conv_2 = ConvBNRelu(name=f'{name}_conv_2', filters=192, kernel_size=3, strides=1, padding='SAME')
        self.max_pool = MaxPool2D(name=f'{name}_max_pool', pool_size=3, strides=2, padding='SAME')

    def call(self, x, **kwargs):
        bridged_input = x
        bridged_input = self.conv(bridged_input)
        bridged_input = self.conv_2(bridged_input)
        bridged_input = self.max_pool(bridged_input)
        return bridged_input
