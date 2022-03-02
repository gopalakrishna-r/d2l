import tensorflow as tf
from d2l import tensorflow as d2l
from tensorpack.models import  Conv2D, BatchNorm,   AvgPooling, FullyConnected, Dropout
from tensorpack.tfutils.argscope import argscope
from tensorflow.keras import Input

def alexnet(input_shape, num_classes):
    input_layer = Input(shape = input_shape, name = "input_layer")
    with argscope([Conv2D], activation='relu',padding='same'), \
            argscope([FullyConnected], units = 4096,  activation='relu'),\
                argscope([MaxPooling], pool_size=3, strides=2):
        x =         Conv2D('conv_1.0', input_layer, padding = 'valid', filters=96, kernel_size=11, strides=4),
        x =         MaxPooling('MaxPool_1.0', x),
        x =         Conv2D('conv_2.0', x, filters=256, kernel_size=5),
        x =         MaxPooling('MaxPool_2.0', x),
        x =         Conv2D('conv_3.0', x, filters=384, kernel_size=3),
        x =         Conv2D('conv_4.0', x, filters=384, kernel_size=3),
        x =         Conv2D('conv_5.0', x, filters=256, kernel_size=3),
        x =         MaxPooling('MaxPool_3.0' , x )
        x =         tf.keras.layers.Flatten()(x)
        x =         FullyConnected('FCN_1.0', x)
        x =         tf.keras.layers.Dropout(0.5)(x)
        x =         FullyConnected('FCN_2.0', x)
        x =         tf.keras.layers.Dropout(0.5) (x)
        
        output_layer =         FullyConnected('FCN_3.0', x , units= num_classes, activation='linear')
        
        return tf.keras.models.Model(input_layer, output_layer)    
        


import numpy as np
from tensorpack.compat import tfv1 as tf  # this should be avoided first in model code

from tensorpack.models.common import layer_register
from tensorpack.models.shape_utils import StaticDynamicShape
from tensorpack.models.tflayer import convert_to_tflayer_args



@layer_register(log_shape=True)
@convert_to_tflayer_args(
    args_names=['pool_size', 'strides'],
    name_mapping={'shape': 'pool_size', 'stride': 'strides'})
def MaxPooling(
        inputs,
        pool_size,
        strides=None,
        padding='valid',
        data_format='channels_last'):
    """
    Same as `tf.layers.MaxPooling2D`. Default strides is equal to pool_size.
    """
    if strides is None:
        strides = pool_size
    layer = tf.keras.layers.MaxPooling2D(pool_size, strides, padding=padding, data_format=data_format)
    ret = layer.apply(inputs)
    return tf.identity(ret, name='output')