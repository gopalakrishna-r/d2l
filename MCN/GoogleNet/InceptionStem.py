import tensorflow as tf
import tf_slim as slim
from tensorflow.python.framework import ops
from tensorflow.python.ops import variable_scope
from tf_slim.layers import utils
from tfutils.argumentscope import conv2d, max_pool2d, dense, global_avg_pool_2d, flatten

BATCH_SIZE = 128


class Inception(tf.keras.layers.Layer):
    def __init__(self, name, c1, c2, c3, c4):
        super().__init__()
        with slim.arg_scope([conv2d, max_pool2d], padding='SAME'), \
                slim.arg_scope([conv2d], activation=tf.nn.relu):
            self.p1_1 = conv2d(name=f'{name}_path_1', filters=c1, kernel_size=1, padding='VALID')

            self.p2_1 = conv2d(name=f'{name}_path_2_1', filters=c2[0], kernel_size=1, padding='VALID')
            self.p2_2 = conv2d(name=f'{name}_path_2_2', filters=c2[1], kernel_size=3)

            self.p3_1 = conv2d(name=f'{name}_path_3_1', filters=c3[0], kernel_size=1)
            self.p3_2 = conv2d(name=f'{name}_path_3_2', filters=c3[1], kernel_size=5)

            self.p4_1 = max_pool2d(name=f'{name}_path_4_1', pool_size=3, strides=1)
            self.p4_2 = conv2d(name=f'{name}_path_4_2', filters=c4, kernel_size=1, padding='VALID')

    def call(self, x):
        p1 = self.p1_1(x)
        p2 = self.p2_2(self.p2_1(x))
        p3 = self.p3_2(self.p3_1(x))
        p4 = self.p4_2(self.p4_1(x))

        return tf.keras.layers.Concatenate()([p1, p2, p3, p4])


class B1(tf.keras.layers.Layer):
    def __init__(self, name):
        super().__init__()
        self.conv = conv2d(name=f'{name}_conv', filters=64, kernel_size=7, strides=2, padding='SAME',
                           activation=tf.nn.relu)
        self.max_pool = max_pool2d(name=f'{name}_max_pool', pool_size=3, strides=2, padding='SAME')

    def call(self, x):
        x = self.conv(x)
        x = self.max_pool(x)
        return x


class B2(tf.keras.layers.Layer):
    def __init__(self, name):
        super().__init__()
        self.conv = conv2d(name=f'{name}_conv', filters=64, kernel_size=1, activation=tf.nn.relu)
        self.conv_2 = conv2d(name=f'{name}_conv_2', filters=192, kernel_size=3, padding='SAME', activation=tf.nn.relu)
        self.max_pool = max_pool2d(name=f'{name}_max_pool', pool_size=3, strides=2, padding='SAME')

    def call(self, x):
        x = self.conv(x)
        x = self.conv_2(x)
        x = self.max_pool(x)
        return x


def build_inception(inputs, channel_1, channel_2, channel_3, channel_4, reuse=None, scope=None):
    with variable_scope.variable_scope(scope, 'inception_stem', [inputs], reuse=reuse) as sc:
        inputs = ops.convert_to_tensor(inputs)
        layer = Inception(sc.name, channel_1, channel_2, channel_3, channel_4)
        outputs = layer.__call__(inputs)

        return utils.collect_named_outputs(None, sc.name, outputs)


def build_configs(block):
    return tuple(
        zip(block['path_1_configs'], block['path_2_configs'], block['path_3_configs'], block['path_4_configs']))


def build_graph():
    inception_block_cofigs = {
        'b3': {'path_1_configs': [64, 128], 'path_2_configs': [(96, 128), (128, 192)],
               'path_3_configs': [(16, 32), (32, 96)], 'path_4_configs': [32, 64]},
        'b4': {'path_1_configs': [192, 160, 128, 112, 256],
               'path_2_configs': [(96, 208), (112, 224), (128, 256), (144, 288), (160, 320)],
               'path_3_configs': [(16, 48), (24, 64), (24, 64), (32, 64), (32, 128)],
               'path_4_configs': [64, 64, 64, 64, 128]},
        'b5': {'path_1_configs': [256, 384], 'path_2_configs': [(160, 320), (192, 384)],
               'path_3_configs': [(32, 128), (48, 128)], 'path_4_configs': [128, 128]},
    }

    input_layer = tf.keras.layers.Input(shape=(96, 96, 1))

    with slim.arg_scope([max_pool2d], pool_size=3, strides=2, padding='SAME'):
        l = B1('B1')(input_layer)
        l = B2('B2')(l)
        l = slim.stack(l, build_inception, build_configs(inception_block_cofigs['b3']), scope='B3_inception')
        l = max_pool2d(name='b3_max_pool')(l)
        l = slim.stack(l, build_inception, build_configs(inception_block_cofigs['b4']), scope='B4_inception')
        l = max_pool2d(name='b4_max_pool')(l)
        l = slim.stack(l, build_inception, build_configs(inception_block_cofigs['b5']), scope='B5_inception')
        l = global_avg_pool_2d(name='gap')(l)
        l = flatten()(l)
        l = dense(name='dense', units=10)(l)
    return tf.keras.Model(inputs=input_layer, outputs=l)
