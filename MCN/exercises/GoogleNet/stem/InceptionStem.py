from tensorflow.keras.layers import Layer,AvgPool2D, MaxPool2D, Concatenate
from tensorflow.keras import Model

from ConvBNRelu import ConvBNRelu


class Stem(Layer):
    def __init__(self, name, c1, c2, c3, c4, use_max_pool):
        super().__init__(name=name)
        self.p1_1 = ConvBNRelu(name=f'{name}_path_1', filters=c1, kernel_size=1, strides=1, padding='VALID')

        self.p2_1 = ConvBNRelu(name=f'{name}_path_2_1', filters=c2[0], kernel_size=1, strides=1, padding='VALID')
        self.p2_2 = ConvBNRelu(name=f'{name}_path_2_2', filters=c2[1], kernel_size=3, strides=1, padding='SAME')

        self.p3_1 = ConvBNRelu(name=f'{name}_path_3_1', filters=c3[0], kernel_size=1,strides=1, padding='VALID')
        self.p3_2 = ConvBNRelu(name=f'{name}_path_3_2', filters=c3[1], kernel_size=3, strides=1, padding='SAME')
        self.p3_3 = ConvBNRelu(name=f'{name}_path_3_3', filters=c3[2], kernel_size=3, strides=1, padding='SAME')

        self.p4_1 = MaxPool2D(name=f'{name}_path_4_1', pool_size=1, strides=1, padding='SAME') \
            if use_max_pool else AvgPool2D(name=f'{name}_path_4_1', pool_size=1, strides=1, padding='SAME')
        self.p4_2 = ConvBNRelu(name=f'{name}_path_4_2', filters=c4, kernel_size=1, strides=1, padding='VALID')

    def call(self, inputs, **kwargs):
        p1 = self.p1_1(inputs)
        p2 = self.p2_2(self.p2_1(inputs))
        p3 = self.p3_3(self.p3_2(self.p3_1(inputs)))
        p4 = self.p4_2(self.p4_1(inputs))
        # print(
        #     f'{self.name} layer input shape {inputs.shape}, output shape {p1.shape}, {p2.shape}, {p3.shape}, {p4.shape}')
        return Concatenate()([p1, p2, p3, p4])

