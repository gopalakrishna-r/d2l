import tensorflow  as tf
from d2l import tensorflow as d2l
from tensorpack.tfutils import argscope
from tensorpack.models import  FullyConnected,  Dropout, MaxPooling, AvgPooling, Conv2D, BatchNorm

def vgg_block(name, num_convs, num_channels, input):
    for i in range(num_convs):
        input = Conv2D(f'{name}_{i}', input, num_channels, kernel_size=3, padding='SAME', activation=tf.nn.relu)
    input = MaxPooling(f'{name}',input, pool_size=2, strides=2)
    return input

def vgg(conv_arch):
    input = tf.keras.Input(shape=(96, 96, 1))
    l = input
    for i, pair in enumerate(conv_arch):
        l = vgg_block(f'layer_{i}_{pair[1]}',pair[0], pair[1], l)

    with argscope([FullyConnected], units = 4096, activation=tf.nn.relu), \
        argscope([Dropout], rate = 0.5):
        #FCN
        l = tf.keras.layers.Flatten()( l)
        l = FullyConnected('fcn_1', l)
        l = Dropout('droput_1', l)
        l = FullyConnected('fcn_2', l)
        l = Dropout('droput_2', l)
    output = FullyConnected('fcn_3', l, units=10, activation='linear')

    return tf.keras.Model(input, output)
    

def vgg16(conv_arch):
    input = tf.keras.Input(shape=(224, 224, 1))
    l = input
    for i, pair in enumerate(conv_arch):
        l = vgg_block(f'layer_{i}_{pair[1]}',pair[0], pair[1], l)

    with argscope([FullyConnected], units = 4096, activation=tf.nn.relu), \
        argscope([Dropout], rate = 0.5):
        #FCN
        l = tf.keras.layers.Flatten()( l)
        l = FullyConnected('fcn_1', l)
        l = FullyConnected('fcn_2', l)
    output = FullyConnected('fcn_3', l, units=10, activation='softmax')

    return tf.keras.Model(input, output)