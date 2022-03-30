import tensorflow as tf
import tf_slim as slim
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.python.framework import ops
from tensorflow.python.ops import variable_scope
from tf_slim.layers import utils

from MCN.exercises.GoogleNet.blocks import B1, B2
from MCN.exercises.GoogleNet.stem import InceptionStem, NaiveInceptionStem
from BatchNormalize import BatchNorm

BATCH_SIZE = 128


# inception net with batch_norm 5x5 conv layers are replaced by two consecutive 3x3 conv layers.
# The number 28×28 inception modules is increased from 2 to 3.
# • Inside the modules, sometimes average, sometimes
# maximum-pooling is employed. This is indicated in

# • There are no across the board pooling layers between
# any two Inception modules, but stride-2 convolution/ pooling layers are employed before the filter
# concatenation in the modules 3c, 4e.


def build_inception(inputs, name, channel_1, channel_2, channel_3, channel_4, reuse=None, scope=None):
    with variable_scope.variable_scope(name, 'inception_stem', [inputs], reuse=reuse) as sc:
        inputs = ops.convert_to_tensor(inputs)
        layer = InceptionStem.Stem(sc.name, channel_1, channel_2, channel_3, channel_4)
        outputs = layer.__call__(inputs)

        return utils.collect_named_outputs(None, sc.name, outputs)


def build_naive_inception(inputs, name, channel_1, channel_2, reuse=None, scope=None):
    with variable_scope.variable_scope(name, 'inception_stem', [inputs], reuse=reuse) as sc:
        inputs = ops.convert_to_tensor(inputs)
        layer = NaiveInceptionStem.Stem(sc.name, channel_1, channel_2)
        outputs = layer.__call__(inputs)

        return utils.collect_named_outputs(None, sc.name, outputs)


def build_configs(block):
    return (
        [(f'inception_stem_{i}', tuple_config[0], tuple_config[1], tuple_config[2], tuple_config[3]) for i, tuple_config
         in
         enumerate(tuple(
             zip(block['path_1_configs'], block['path_2_configs'], block['path_3_configs'], block['path_4_configs'])))])


def build_naive_inception_configs(block):
    return 'inception_stem',


def build_graph():
    inception_block_cofigs = {
        '3a_3b': {'path_1_configs': [64, 64], 'path_2_configs': [(64, 64), (64, 96)],
                  'path_3_configs': [(64, 96, 96), (64, 96, 96)], 'path_4_configs': [32, 64]},
        '3c': {'path_2_configs': [128, 160], 'path_3_configs': [64, 96, 96]},
        '4a_4b_4c_4d': {'path_1_configs': [224, 192, 160, 96],
                        'path_2_configs': [(64, 96), (96, 128), (128, 160), (128, 192)],
                        'path_3_configs': [(96, 128, 128), (96, 128, 128), (128, 160, 160),
                                           (192, 256, 256)],
                        'path_4_configs': [128, 128, 128, 128]},
        '4e': {'name': 'inception_stem', 'path_2_configs': [128, 192], 'path_3_configs': [192, 256, 256]},
        '5a_5b': {'name': 'inception_stem', 'path_1_configs': [352, 352], 'path_2_configs': [(192, 320), (192, 320)],
                  'path_3_configs': [(160, 224, 224), (192, 224, 224)], 'path_4_configs': [128, 128]},
    }

    input_layer = tf.keras.layers.Input(shape=(224, 224, 1))

    bridged_input = B1.InceptionBlk1('B1')(input_layer)
    bridged_input = B2.InceptionBlk2('B2')(bridged_input)
    bridged_input = slim.stack(bridged_input, build_inception, build_configs(inception_block_cofigs['3a_3b']),
                               scope='inception_blk_3')
    bridged_input = build_naive_inception(bridged_input, 'inception_blk_3c',
                                          inception_block_cofigs['3c']['path_2_configs'],
                                          inception_block_cofigs['3c']['path_3_configs'])

    bridged_input = slim.stack(bridged_input, build_inception, build_configs(inception_block_cofigs['4a_4b_4c_4d']),
                               scope='inception_blk_4')
    bridged_input = build_naive_inception(bridged_input, 'inception_blk_4_e',
                                          inception_block_cofigs['4e']['path_2_configs'],
                                          inception_block_cofigs['4e']['path_3_configs'])
    config = build_configs(inception_block_cofigs['5a_5b'])
    # bridged_input = build_inception(bridged_input, channel_1, channel_2, channel_3, channel_4, reuse=None, scope=None)
    bridged_input = Flatten()(bridged_input)
    bridged_input = Dense(name='dense', units=10)(bridged_input)
    return tf.keras.Model(inputs=input_layer, outputs=bridged_input)

def net():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=6, kernel_size=5,
                               input_shape=(28, 28, 1)),
        BatchNorm(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=5),
        BatchNorm(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120),
        BatchNorm(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.Dense(84),
        BatchNorm(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.Dense(10)]
    )