import tensorflow as tf
import tf_slim as slim
from keras_tuner import HyperParameters
from tensorflow.python.framework import ops
from tensorflow.python.ops import variable_scope
from tf_slim.layers import utils
from tfutils.argumentscope.SlimUtils import reshape, conv2d, max_pool2d, dropout, globalAveragePooling2D, flatten

BATCH_SIZE = 128


def build_graph(hp: HyperParameters):
    input_layer = tf.keras.Input(shape=(224, 224, 1), name='input')
    l = input_layer

    filters_1 = hp.Choice(name='filters_1', values=[90, 92, 94, 96, 100])
    filters_2 = hp.Choice(name='filters_2', values=[64, 128, 256, 512])
    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    optimizer = tf.keras.optimizers.SGD(learning_rate)

    channel_configs = [filters_1, filters_2, 384, 10]
    kernel_size_configs = [11, 5, 3, 1]
    strides_configs = [4, 1, 1, 1]
    padding_configs = ['VALID', 'SAME', 'SAME', 'SAME']
    max_pool_configs = [True, True, True, False]
    nin_stem_configs = tuple(
        zip(channel_configs, kernel_size_configs, strides_configs, padding_configs, max_pool_configs))

    with slim.arg_scope([dropout], rate=0.5):
        l = slim.stack(l, nin_block,
                       [(f'nin_block_{i}', tuple[0], tuple[1], tuple[2], tuple[3], tuple[4]) for i, tuple in
                        enumerate(nin_stem_configs[:-1])])
        l = dropout(name='dropout')(l)
        l = slim.stack(l, nin_block, [(f'nin_block_{len(nin_stem_configs) - 1}', nin_stem_configs[-1][0],
                                       nin_stem_configs[-1][1], nin_stem_configs[-1][2], nin_stem_configs[-1][3],
                                       nin_stem_configs[-1][4])])
        l = globalAveragePooling2D(name='gap')(l)
        l = reshape(name='reshape', target_shape=(1, 1, 10))(l)
        logits = flatten(name='flatten')(l)

    net = tf.keras.Model(inputs=input_layer, outputs=logits)

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    net.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    return net


@slim.add_arg_scope
def nin_block(inputs, name, num_channels, kernel_size, strides, padding, enable_max_pool=True, reuse=None, scope=None):
    with variable_scope.variable_scope(
            scope,
            'nin_stem', [inputs],
            reuse=reuse) as sc:
        inputs = ops.convert_to_tensor(inputs)
        layer = NinStem(name, num_channels, kernel_size, strides, padding, enable_max_pool)
        outputs = layer.__call__(inputs)

        return utils.collect_named_outputs(None, sc.name, outputs)


class NinStem(tf.keras.layers.Layer):
    def __init__(self, name, num_channels, kernel_size, strides, padding, enable_max_pool):
        super().__init__(name=name)
        self.conv_1 = conv2d(name=f'{name}_conv1', filters=num_channels, kernel_size=kernel_size, strides=strides,
                             padding=padding, activation='relu')
        self.conv_2 = conv2d(name=f'{name}_conv2', filters=num_channels, kernel_size=1, strides=1, padding='SAME',
                             activation='relu')
        self.conv_3 = conv2d(name=f'{name}_conv3', filters=num_channels, kernel_size=1, strides=1, padding='SAME',
                             activation='relu')
        self.max_pool = max_pool2d(name=f'{name}_max_pool', pool_size=3, strides=2)
        self.enable_max_pool = enable_max_pool

    def call(self, l, **kwargs):
        l = self.conv_1(l)
        l = self.conv_2(l)
        l = self.conv_3(l)
        if self.enable_max_pool:
            l = self.max_pool(l)
        return l
