
import argparse
import os
import sys
from ast import parse

import tensorflow as tf
from tensorpack.callbacks import (ClassificationError, InferenceRunner,
                                  ModelSaver, ScalarStats,
                                  ScheduledHyperParamSetter)
from tensorpack.dataflow import (AugmentImageComponent, BatchData, dataset,
                                 imgaug)
from tensorpack.dataflow.regularize import l2_regularizer, regularize_cost
from tensorpack.graph_builder.model_desc import ModelDesc
from tensorpack.models import (AvgPooling, BatchNorm, Conv2D, Dropout, Flatten,
                               FullyConnected, GlobalAvgPooling, MaxPooling,
                               nonlin)
from tensorpack.tfutils.argscope import argscope
from tensorpack.tfutils.common import get_global_step_var
from tensorpack.tfutils.sessinit import SmartInit
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary
from tensorpack.train import SimpleTrainer, launch_train_with_config
from tensorpack.train.config import TrainConfig
from tensorpack.utils import logger
from tensorpack.input_source import QueueInput

from block import vgg_block

BATCH_SIZE = 32

class Model(ModelDesc):
    def __init__(self, n):
        super(Model, self).__init__()
        self.n = n

    def inputs(self):
        return [tf.TensorSpec([None, 224, 224], tf.float32, 'input'),
                tf.TensorSpec([None], tf.int32, 'label')]

    def build_graph(self, image, label):
        image = image / 128.0
        l = tf.expand_dims(image, 3) * 2 - 1
        assert tf.test.is_gpu_available()

        def vgg_block(name, num_convs, num_channels, input):
            with tf.compat.v1.variable_scope(name):
                for i in range(num_convs):
                    input = Conv2D(f'{name}_conv_{i+1}.0', input, num_channels)
                input = MaxPooling('maxpooling_1.0',input, pool_size=2, strides=2)
            return input

        with argscope([FullyConnected], units = 4096), \
          argscope([FullyConnected, Conv2D], activation=tf.nn.relu), \
            argscope([Conv2D, MaxPooling], data_format = 'channels_last'), \
              argscope([Conv2D], kernel_size=3, padding = 'SAME'), \
                argscope([Dropout], rate = 0.5):
                    convo_arch = [(2, 64), (2, 128), (4, 256), (4, 512), (4, 512)]
                    for i, pair in enumerate(convo_arch):
                        l = vgg_block(f'layer_{i}_{pair[1]}',pair[0], pair[1], l)
                #FCN
                    l = Flatten('flatten', l)
                    l = FullyConnected('fcn_1', l)
                    l = FullyConnected('fcn_2', l)
        logits = FullyConnected('fcn_3', l, units=10, activation = tf.nn.softmax)
                
        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')

        correct = tf.cast(tf.nn.in_top_k(predictions=logits, targets=label, k=1), tf.float32, name='correct')
        # monitor training error
        add_moving_summary(tf.reduce_mean(correct, name='accuracy'))

        # weight decay on all W of fc layers
        wd_cost = regularize_cost('fc.*/W', l2_regularizer(4e-4), name='regularize_loss')
        add_moving_summary(cost, wd_cost)

        return tf.add_n([cost, wd_cost], name='cost')
    
    def optimizer(self):
        lr = tf.compat.v1.get_variable(
            'learning_rate', initializer=0.1, trainable=False)
        return tf.compat.v1.train.AdamOptimizer(lr, 0.9)


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
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma seperated list of GPU(s) to use.')
    parser.add_argument(
        '-n', '--num-units', help='number of units in each stage', type=int, default=5)
    parser.add_argument('--load', help='load model for training')
    parser.add_argument('--logdir', help='log directory')
    args = parser.parse_args()
   
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.logdir:
        logger.set_logger_dir(args.logdir)
    else:
        logger.auto_set_dir()
    dataset_train = get_data('train')
    dataset_test = get_data('test')
    config = TrainConfig(
        model=Model(n=args.num_units),
        dataflow=QueueInput(dataset_train),
        callbacks=[
            ModelSaver(),   # save the model after every epoch
            InferenceRunner(    # run inference(for validation) after every epoch
                dataset_test,   # the DataFlow instance used for validation
                ScalarStats(    # produce `val_accuracy` and `val_cross_entropy_loss`
                    ['cost', 'accuracy'], prefix='val')),
            # MaxSaver needs to come after InferenceRunner to obtain its score
            ScheduledHyperParamSetter(
                'learning_rate', [(1, 0.1), (32, 0.01), (48, 0.001)])
        ],
        steps_per_epoch=1000,
        max_epoch=5,
        session_init=SmartInit(args.load),
    )
    launch_train_with_config(config, SimpleTrainer())
