from warnings import filters
import tensorflow as tf
from tfutils.argumentscope.SlimUtils import flatten, dense, conv2d, max_pool2d, dropout
from tf_slim import arg_scope, repeat
from tensorflow.keras import Input

def alexnet():
    image = Input(shape=(224, 224, 1))
    with arg_scope([conv2d, dense], activation=tf.nn.relu), \
            arg_scope([conv2d], kernel_size=3, padding='SAME'), \
                arg_scope([max_pool2d], pool_size=3, strides=2, padding='VALID'), \
                    arg_scope([dense], units = 4096,  activation='relu'):
        
        l = conv2d(name = 'conv1', filters=96, kernel_size=11, strides=4, padding='VALID')(image)
        l = max_pool2d(name = 'pool1')(l)

        l = conv2d(name = 'conv2',  filters=256, kernel_size=5)(l)
        l = max_pool2d(name = 'pool2' )(l)

        l = conv2d(name = 'conv3' , filters = 384)(l)
        l = conv2d(name = 'conv4',  filters = 384) (l)
        l = conv2d(name = 'conv5',  filters=256)(l)
        l = max_pool2d(name = 'pool3' )(l)
        l = flatten(name = 'flatten' )(l)
        l = dense(name = 'fc1' )(l)
        l = dropout(name = 'droput1', rate = 0.5) (l)
        l = dense(name = 'fc2' )(l)
        l = dropout(name = 'dropout2', rate = 0.5)(l)
        
        
    output_layer =         dense(name = 'FCN3', units= 10, activation='linear')( l )
        
    return tf.keras.models.Model(image, output_layer)    
        

