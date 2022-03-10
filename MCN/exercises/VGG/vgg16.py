
import tensorflow as tf
import tf_slim as slim

    
from tfutils.argumentscope.SlimUtils import flatten, dense, conv2d, max_pool2d, dropout

tfl = tf.keras.layers

BATCH_SIZE = 16

def vgg16(convo_arch):
  input = tf.keras.Input(shape=(96, 96, 1))
  l = input
  with slim.arg_scope([dense], units = 4096),\
   slim.arg_scope([conv2d, dense], activation=tf.nn.relu),\
    slim.arg_scope([dropout], rate = 0.5):
      for i, pair in enumerate(convo_arch):
          l = vgg_block(f'layer_{i}_{pair[1]}',pair[0], pair[1], l)
      l = flatten(name = 'Flatten')(l)
      l = dense( name = 'fcn_1')(l)
      l = dropout(name = 'droput_1')(l)
      l = dense(name = 'fcn_2')(l)
      l = dropout(name = 'droput_2')(l)
      logits = dense( units=10, name = 'fcn_3')(l)
      return tf.keras.Model(input, logits)


def vgg_block(name, num_convs, num_channels, l):
        for _ in range(num_convs):
            l = conv2d(name = f'{name}_conv', filters = num_channels, kernel_size=3, padding = 'SAME')(l)
        l = max_pool2d(f'{name}_max_pool', pool_size=2, strides=2)(l)
        return l
    
              
