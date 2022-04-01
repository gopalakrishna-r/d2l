from ConvBNRelu import ConvBNRelu
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import MaxPool2D, Concatenate


class Stem(Layer):
    def __init__(self, name, c1, c2):
        super().__init__(name=name)
        self.p1_1 = ConvBNRelu(name=f'{name}_path_1_1', filters=c1[0], kernel_size=1, strides=1, padding='VALID')
        self.p1_2 = ConvBNRelu(name=f'{name}_path_1_2', filters=c1[1], kernel_size=3, strides=2, padding='SAME')

        self.p2_1 = ConvBNRelu(name=f'{name}_path_2_1', filters=c2[0], kernel_size=1, strides=1, padding='VALID')
        self.p2_2 = ConvBNRelu(name=f'{name}_path_2_2', filters=c2[1], kernel_size=3, strides=1, padding='SAME')
        self.p2_3 = ConvBNRelu(name=f'{name}_path_2_3', filters=c2[2], kernel_size=3, strides=2, padding='SAME')

        self.p3_1 = MaxPool2D(name=f'{name}_path_3_1', pool_size=3, strides=2, padding='SAME')

    def call(self, inputs, **kwargs):
        p1 = self.p1_2(self.p1_1(inputs))

        p2 = self.p2_3(self.p2_2(self.p2_1(inputs)))
        p3 = self.p3_1(inputs)
        # print(
        #     f'{self.name} layer input shape {inputs.shape}, output shape {p1.shape}, {p2.shape}, {p3.shape}')
        return Concatenate()([p1, p2, p3])
