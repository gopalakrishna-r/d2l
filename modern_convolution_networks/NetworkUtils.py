import tensorflow as tf
from d2l import tensorflow as d2l
from tensorpack.models import  Conv2D, BatchNorm,MaxPooling, AvgPooling, FullyConnected,  Flatten, Dropout
from tensorpack.tfutils.argscope import argscope
from tensorflow.keras import Input
from tensorpack.tfutils.common import get_tf_version_tuple
from tensorpack.tfutils.tower import get_current_tower_context
from tensorpack.utils.argtools import graph_memoized
from tensorpack.models.common import layer_register

def alexnet():
    image = Input(shape=(224, 224, 1))
    with argscope([Conv2D, FullyConnected], activation=tf.nn.relu), \
            argscope([Conv2D], kernel_size=3, padding='SAME'), \
                argscope([MaxPooling], pool_size=3, strides=2, padding='VALID'), \
                    argscope([FullyConnected], units = 4096,  activation='relu'):
        
        l = Conv2D('conv1', image, filters=96, kernel_size=11, strides=4, padding='VALID')
        l = MaxPooling('pool1', l)

        l = Conv2D('conv2', l, filters=256, kernel_size=5)
        l = MaxPooling('pool1', l)

        l = Conv2D('conv3', l, filters=384)
        l = Conv2D('conv4', l, filters=384)
        l = Conv2D('conv5', l, filters=256)
        l = MaxPooling('pool1', l)
        l = Flatten('flatten', l)
        l = FullyConnected('fc1', l)
        l = Dropout('droput1', l , 0.5) # 0.5 by defualt
        l = FullyConnected('fc2', l)
        l = Dropout('dropout2', l,0.5)
        
        
    output_layer =         FullyConnected('FCN3', l , units= 10, activation='linear')
        
    return tf.keras.models.Model(image, output_layer)    
        

