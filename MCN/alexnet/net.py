import tensorflow as tf
from tensorflow.keras import Input
from tf_slim import arg_scope
from tfutils.argumentscope.SlimUtils import flatten, dense, conv2d, max_pool2d, dropout


def alexnet():
    image = Input(shape=(224, 224, 1))
    with arg_scope([conv2d, dense], activation=tf.nn.relu), \
            arg_scope([conv2d], kernel_size=3, padding='SAME'), \
            arg_scope([max_pool2d], pool_size=3, strides=2, padding='VALID'), \
            arg_scope([dense], units=4096, activation='relu'):
        conv_input = conv2d(name='conv1', filters=96, kernel_size=11, strides=4, padding='VALID')(image)
        conv_input = max_pool2d(name='pool1')(conv_input)

        conv_input = conv2d(name='conv2', filters=256, kernel_size=5)(conv_input)
        conv_input = max_pool2d(name='pool2')(conv_input)

        conv_input = conv2d(name='conv3', filters=384)(conv_input)
        conv_input = conv2d(name='conv4', filters=384)(conv_input)
        conv_input = conv2d(name='conv5', filters=256)(conv_input)
        conv_input = max_pool2d(name='pool3')(conv_input)
        conv_input = flatten(name='flatten')(conv_input)
        conv_input = dense(name='fc1')(conv_input)
        conv_input = dropout(name='droput1', rate=0.5)(conv_input)
        conv_input = dense(name='fc2')(conv_input)
        conv_input = dropout(name='dropout2', rate=0.5)(conv_input)

    output_layer = dense(name='FCN3', units=10, activation='linear')(conv_input)

    return tf.keras.models.Model(image, output_layer)
