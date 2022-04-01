import tensorflow as tf
import tf_slim as slim
from tensorflow.keras.layers import Flatten, Dense, AvgPool2D
from tensorflow.python.framework import ops
from tensorflow.python.ops import variable_scope
from tf_slim.layers import utils

from blocks import B1, B2
from stem import InceptionStem, NaiveInceptionStem

BATCH_SIZE = 128


# inception net with batch_norm 5x5 conv layers are replaced by two consecutive 3x3 conv layers.
# The number 28×28 inception modules is increased from 2 to 3.
# • Inside the modules, sometimes average, sometimes
# maximum-pooling is employed. This is indicated in

# • There are no across the board pooling layers between
# any two Inception modules, but stride-2 convolution/ pooling layers are employed before the filter
# concatenation in the modules 3c, 4e.


def build_inception(inputs, name, channel_1, channel_2, channel_3, channel_4, use_max_pool=False, reuse=None,
                    scope=None):
    with variable_scope.variable_scope(name, 'inception_stem', [inputs], reuse=reuse) as sc:
        inputs = ops.convert_to_tensor(inputs)
        layer = InceptionStem.Stem(sc.name, channel_1, channel_2, channel_3, channel_4, use_max_pool)
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
             zip(block['path_1_configs'], block['path_2_configs'], block['path_3_configs'], block['path_4_configs'],
                 block['max_pool_configs'])))])


def build_naive_inc_configs(block):
    return [('naive_inception_stem', block['path_2_configs'], block['path_3_configs'])]


def build_graph():
    inception_block_cofigs = {
        '3a_3b': {'path_1_configs': [64, 64], 'path_2_configs': [(64, 64), (64, 96)],
                  'path_3_configs': [(64, 96, 96), (64, 96, 96)], 'path_4_configs': [32, 64],
                  'max_pool_configs': [False, False]},
        '3c': {'path_2_configs': [128, 160], 'path_3_configs': [64, 96, 96]},
        '4a_4b_4c_4d': {'path_1_configs': [224, 192, 160, 96],
                        'path_2_configs': [(64, 96), (96, 128), (128, 160), (128, 192)],
                        'path_3_configs': [(96, 128, 128), (96, 128, 128), (128, 160, 160),
                                           (160, 192, 192)],
                        'path_4_configs': [128, 128, 128, 128], 'max_pool_configs': [False, False, False, False]},
        '4e': {'path_2_configs': [128, 192], 'path_3_configs': [192, 256, 256]},
        '5a_5b': {'path_1_configs': [352, 352], 'path_2_configs': [(192, 320), (192, 320)],
                  'path_3_configs': [(160, 224, 224), (192, 224, 224)], 'path_4_configs': [128, 128],
                  'max_pool_configs': [False, True]}
    }

    input_layer = tf.keras.layers.Input(shape=(224, 224, 1))

    bridged_input = B1.Blk1('B1')(input_layer)
    bridged_input = B2.Blk2('B2')(bridged_input)
    bridged_input = slim.stack(bridged_input, build_inception, build_configs(inception_block_cofigs['3a_3b']),
                               scope='inception_blk_3')
    bridged_input = slim.stack(bridged_input, build_naive_inception,
                               build_naive_inc_configs(inception_block_cofigs['3c']), scope='inception_blk_3c')

    bridged_input = slim.stack(bridged_input, build_inception, build_configs(inception_block_cofigs['4a_4b_4c_4d']),
                               scope='inception_blk_4')
    bridged_input = slim.stack(bridged_input, build_naive_inception,
                               build_naive_inc_configs(inception_block_cofigs['4e']), scope='inception_blk_4_e')
    bridged_input = slim.stack(bridged_input, build_inception, build_configs(inception_block_cofigs['5a_5b']),
                               scope='inception_blk_5')
    bridged_input = AvgPool2D(pool_size=7, strides=1, name='AvgPool')(bridged_input)
    bridged_input = Flatten(name = 'flatten')(bridged_input)
    bridged_input = Dense(name='dense', units=10)(bridged_input)
    return tf.keras.Model(inputs=input_layer, outputs=bridged_input)
