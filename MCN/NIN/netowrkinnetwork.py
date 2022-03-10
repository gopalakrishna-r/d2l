import os
import sys

from tfutils.argumentscope.SlimUtils import reshape, conv2d, max_pool2d, dropout, globalAveragePooling2D, flatten
import tensorflow as tf
import tf_slim as slim



BATCH_SIZE = 128

@slim.add_arg_scope
def nin_block(**kwargs):
  l , name ,num_channels, kernel_size, strides, padding = kwargs['inputs'], kwargs['name'], kwargs['num_channels'], kwargs['kernel_size'], kwargs['strides'], kwargs['padding']
  with tf.compat.v1.variable_scope(name):
    l = conv2d(name = f'{name}_conv1',  filters = num_channels, kernel_size = kernel_size, strides = strides, padding = padding)(l)
    l = conv2d(name = f'{name}_conv2',  filters = num_channels, kernel_size = 1)(l)
    l = conv2d(name = f'{name}_conv3',  filters = num_channels, kernel_size = 1)(l)
    return l


def build_graph():
        input_layer = tf.keras.Input(shape=(224, 224, 1), name='input')
        l = input_layer
        with slim.arg_scope([max_pool2d], pool_size = 3, strides = 2), \
            slim.arg_scope([nin_block], strides=1, padding='SAME'),\
                slim.arg_scope([conv2d], kernel_size=3, padding = 'SAME', activation = 'relu'), \
                    slim.arg_scope([dropout], rate = 0.5):
                l = nin_block(
                    name='nin_block1.0',
                    inputs=l,
                    num_channels=96,
                    kernel_size=11,
                    strides=4,
                    padding='VALID',
                )
                l =   max_pool2d(name = 'pool1')(l)
                l = nin_block(
                    name='nin_block2.0',
                    inputs=l,
                    num_channels=256,
                    kernel_size=5,
                )
                l =   max_pool2d(name = 'pool2')(l)
                l = nin_block(
                    name='nin_block3.0',
                    inputs=l,
                    num_channels=384,
                    kernel_size=3,
                )
                l =   max_pool2d(name = 'pool3')(l)
                l =   dropout(name ='dropout')(l)
                l = nin_block(
                    name='nin_block4.0',
                    inputs=l,
                    num_channels=10,
                    kernel_size=3,
                )
                l =   globalAveragePooling2D(name = 'gap')(l)
                l =   reshape(name = 'reshape',target_shape = (1,1,10) )(l)
                logits =  flatten(name = 'flatten')(l)

        return tf.keras.Model(inputs=input_layer, outputs=logits)  

           
