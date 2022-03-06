
import os
import sys

from tensorpack.tfutils.summary import add_moving_summary, add_param_summary
from tensorpack.tfutils.common import get_global_step_var
from tensorpack.models import (AvgPooling, BatchNorm, Conv2D, GlobalAvgPooling,
                               nonlin)
from tensorpack.tfutils.argscope import argscope
from tensorpack.tfutils.sessinit import SmartInit
from tensorpack.dataflow.regularize import regularize_cost
from tensorpack.graph_builder.model_desc import ModelDesc
from tensorpack.utils import logger
from tensorpack.dataflow import dataset, imgaug, BatchData, AugmentImageComponent
from tensorpack.train.config import TrainConfig
from tensorpack.train import launch_train_with_config, SimpleTrainer
from tensorpack.callbacks import ModelSaver, InferenceRunner, ScheduledHyperParamSetter, ScalarStats, ClassificationError
from tensorpack.models import Conv2D, FullyConnected,  Dropout, MaxPooling
from block import vgg_block
import argparse
from ast import parse



import tensorflow as tf

BATCH_SIZE = 32

class Model(ModelDesc):
    def __init__(self, n):
        super(Model, self).__init__()
        self.n = n

    def inputs(self):
        return [tf.TensorSpec([None, 224, 224,1], tf.float32, 'input'),
                tf.TensorSpec([None], tf.int32, 'label')]

    def build_graph(self, image, label):
        image = image / 128.0
        assert tf.test.is_gpu_available()

        def vgg_block(name, num_convs, num_channels, input):
            with tf.compat.v1.variable_scope(name):
                with argscope(Conv2D, kernel_shape=3, padding = 'SAME',activation = tf.nn.relu, data_format = 'channels_last'):
                    for i in range(num_convs):
                        input = Conv2D(f'conv_{i+1}.0', input, num_channels)
                    input = MaxPooling('maxpooling_1.0',input, pool_size=2, strides=2)
                return input

        with argscope([FullyConnected], units = 4096, activation=tf.nn.relu), \
            argscope([Dropout], rate = 0.5):
                convo_arch = [(2, 64), (2, 128), (3, 256), (3, 512), (3, 512)]
                for i, pair in enumerate(convo_arch):
                    l = vgg_block(f'layer_{i}_{pair[1]}',pair[0], pair[1], image)
                #FCN
                l = tf.keras.layers.Flatten()(l)
                l = FullyConnected('fcn_1', l)
                l = FullyConnected('fcn_2', l)
        output = FullyConnected('fcn_3', l, units=10)
                
        tf.nn.softmax(output, name='output')
        
        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=output)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')
        wrong = tf.cast(tf.logical_not(tf.nn.in_top_k(label, output, 1)), tf.float32,  name='wrong_vector')
        
        add_moving_summary(tf.reduce_mean(wrong, name='train_error'))
        wd_w = tf.compat.v1.train.exponential_decay(
            0.0002, get_global_step_var(), 480000, 0.2, True)
        wd_cost = tf.multiply(wd_w, regularize_cost(
            '.*/W', tf.nn.l2_loss), name='wd_cost')
        add_moving_summary(cost, wd_cost)
        add_param_summary(('.*/W', ['histogram']))
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
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
    # Invalid device or cannot modify virtual devices once initialized.
        pass
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
        dataflow=dataset_train,
        callbacks=[
            ModelSaver(),
            InferenceRunner(dataset_test, [
                ScalarStats('cost'), ClassificationError('wrong_vector')
            ]),
            ScheduledHyperParamSetter(
                'learning_rate', [(1, 0.1), (32, 0.01), (48, 0.001)])
        ],
        steps_per_epoch=1000,
        max_epoch=64,
        session_init=SmartInit(args.load),
    )
    launch_train_with_config(config, SimpleTrainer())
