
import tensorflow as tf
import tf_slim as slim
from tfutils.argumentscope.SlimUtils import flatten, dense, conv2d, max_pool2d, dropout
from  tf_slim.layers import utils
from tensorflow.python.framework import ops
from tensorflow.python.ops import variable_scope

tfl = tf.keras.layers

BATCH_SIZE = 16

def vgg16(convo_arch):
  input = tf.keras.Input(shape=(96, 96, 1))
  l = input
  with slim.arg_scope([dense], units = 4096 , activation=tf.nn.relu),\
    slim.arg_scope([dropout], rate = 0.5):
      l = slim.stack(l, vgg_block,[(f'layer_{i}_{pair[1]}',pair[0], pair[1]) for i, pair in enumerate(convo_arch)])
      l = flatten(name = 'Flatten')(l)
      l = dense( name = 'fcn_1')(l)
      l = dropout(name = 'droput_1')(l)
      l = dense(name = 'fcn_2')(l)
      l = dropout(name = 'droput_2')(l)
      logits = dense( units=10, activation = tf.identity, name = 'fcn_3')(l)
      return tf.keras.Model(input, logits)



class vgg_stem(tf.keras.Model):
  def __init__(self, name, num_convs, num_channels):
      super().__init__(name = name)
      self.conv_layers  = [conv2d(name = f'{self.name}_conv', filters = num_channels,
                       kernel_size=3, padding = 'SAME', activation = tf.nn.relu) for _ in range(num_convs)]
      self.max_pool = max_pool2d(name = f'{self.name}_max_pool', pool_size=2, strides=2)
      
  def call(self, l):
    for layer in self.conv_layers:
      l = layer(l)
    l = self.max_pool(l)
    return l


@slim.add_arg_scope
def vgg_block(inputs ,name, num_convs, num_channels, reuse=None, scope=None):
  with variable_scope.variable_scope(
      scope,
      'vgg_stem', [inputs],
      reuse=reuse) as sc:
    inputs = ops.convert_to_tensor(inputs)
    layer = vgg_stem(name = name, num_convs = num_convs, num_channels = num_channels)
    outputs = layer.__call__(inputs)

    return utils.collect_named_outputs(None, sc.name, outputs)
    