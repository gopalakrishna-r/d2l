from tensorflow.keras.layers import MaxPool2D, Concatenate, Dense, Layer

from ..ConvBNRelu import ConvBNRelu


class InceptionBlk1(Layer):
    def __init__(self, name):
        super().__init__()
        self.conv = ConvBNRelu(name=f'{name}_conv', filters=64, kernel_size=7, strides=2)
        self.max_pool = MaxPool2D(name=f'{name}_max_pool', pool_size=3, strides=2, padding='SAME')

    def call(self, x, **kwargs):
        bridged_input = x
        bridged_input = self.conv(bridged_input)
        bridged_input = self.max_pool(bridged_input)
        return bridged_input
