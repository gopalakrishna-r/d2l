
import argparse
import os
import sys
from ast import parse

import tensorflow as tf
from tensorpack.callbacks import ( InferenceRunner,
                                  ModelSaver, ScalarStats,StatMonitorParamSetter
                                  )
from tensorpack.dataflow import (AugmentImageComponent, BatchData, dataset,
                                 imgaug)
from tensorpack.dataflow.regularize import regularize_cost, l2_regularizer
from tensorpack.graph_builder.model_desc import ModelDesc
from tensorpack.models import ( Conv2D, Dropout, Flatten,
                               GlobalAvgPooling, MaxPooling, nonlin)
from tensorpack.tfutils.argscope import argscope
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary
from tensorpack.train import SimpleTrainer, launch_train_with_config
from tensorpack.train.config import TrainConfig
from tensorpack.utils import logger

BATCH_SIZE = 64

class Model(ModelDesc):
    def __init__(self, n):
        super(Model, self).__init__()
        self.n = n

    def inputs(self):
        return [tf.TensorSpec([None, 224, 224], tf.float32, 'input'),
                tf.TensorSpec([None], tf.int32, 'label')]

    def build_graph(self, image, label):
        image = image / 128.0
        assert tf.test.is_gpu_available()
        l = tf.expand_dims(image, 3) * 2 - 1

        def nin_block(name, l, num_channels, kernel_size, strides, padding):
            with tf.compat.v1.variable_scope(name):
              with argscope(Conv2D, activation = 'relu'):
                  l = Conv2D('conv1', l, num_channels, kernel_size = kernel_size, strides = strides, padding = padding)
                  l = Conv2D('conv2', l, num_channels, kernel_size = 1)
                  l = Conv2D('conv3', l, num_channels, kernel_size = 1)
                  return l
                  
               
        with argscope([MaxPooling], pool_size = 3, strides = 2), \
              argscope([Conv2D], kernel_size=3, padding = 'SAME'), \
                argscope([Dropout], rate = 0.5):
                    l =   nin_block(f'nin_block1.0', l, 96, kernel_size=11, strides=4, padding='VALID')
                    l =   MaxPooling('pool1', l)
                    l =   nin_block(f'nin_block2.0', l, 256, kernel_size= 5,  strides=1, padding='SAME')    
                    l =   MaxPooling('pool2', l)
                    l =   nin_block(f'nin_block3.0', l, 384, kernel_size= 3,  strides=1, padding='SAME')
                    l =   MaxPooling('pool3', l)
                    l =   Dropout('dropout', l)
                    l =   nin_block(f'nin_block4.0', l, 10, kernel_size= 3,  strides=1, padding='SAME')
                    l =   GlobalAvgPooling('gap', l)
                    l =   tf.keras.layers.Reshape((1, 1, 10))(l)
                    logits =  Flatten('flatten', l)
                
                
        
     # build cost function by tensorflow
        cost = tf.nn.weighted_cross_entropy_with_logits(logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')

        correct = tf.cast(tf.nn.in_top_k(predictions=logits, targets=label, k=1), tf.float32, name='correct')
        # monitor training error
        add_moving_summary(tf.reduce_mean(correct, name='accuracy'))

        # weight decay on all W of fc layers
        wd_cost = regularize_cost('fc.*/W', l2_regularizer(4e-4), name='regularize_loss')
        add_moving_summary(cost, wd_cost)

        add_param_summary(('.*/W', ['histogram']))   # monitor W
        return tf.add_n([cost, wd_cost], name='cost')


    def optimizer(self):
        lr = tf.compat.v1.get_variable(
            'learning_rate', initializer=0.1, trainable=False)
        return tf.compat.v1.train.MomentumOptimizer(lr, 0.9)



def get_data(train_or_test):
    isTrain = train_or_test == 'train'
    ds = dataset.FashionMnist(train_or_test)
    augmentors = [
        imgaug.Resize((224, 224)),
    ]
    ds = AugmentImageComponent(ds, augmentors)
    ds = BatchData(ds, BATCH_SIZE, remainder=not isTrain)

    return ds


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    logger.auto_set_dir()
    dataset_train = get_data('train')
    dataset_test = get_data('test')
    
    def lr_func(lr):
        if lr < 3e-5:
            raise StopTraining()
        return lr * 0.31
    
    cfg = TrainConfig(
        model=Model(10),
        dataflow=dataset_train,
        callbacks=[
            ModelSaver(),
            InferenceRunner(
                dataset_test,
                ScalarStats(['cross_entropy_loss', 'accuracy'])),
                StatMonitorParamSetter('learning_rate', 'validation_accuracy', lr_func,
                                   threshold=0.001, last_k=10, reverse=True),
        ],
        max_epoch=5,
    )
    launch_train_with_config(cfg, SimpleTrainer())
    