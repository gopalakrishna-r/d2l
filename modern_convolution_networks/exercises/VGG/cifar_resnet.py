
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
from tensorpack.models import Conv2D, FullyConnected,  BNReLU
import argparse
from ast import parse



import tensorflow as tf

BATCH_SIZE = 64

class Model(ModelDesc):
    def __init__(self, n):
        super(Model, self).__init__()
        self.n = n

    def inputs(self):
        return [tf.TensorSpec([None, 32, 32, 3], tf.float32, 'input'),
                tf.TensorSpec([None], tf.int32, 'label')]

    def build_graph(self, image, label):
        image = image / 128.0
        assert tf.test.is_gpu_available()
        image = tf.transpose(image, [0, 3, 1, 2])

        def residual(name, l, increase_dim=False, first=False):
            shape = l.get_shape().as_list()
            in_channel = shape[1]
            if increase_dim:
                out_channel = in_channel * 2
                stride1 = 2
            else:
                out_channel = in_channel
                stride1 = 1
            with tf.compat.v1.variable_scope(name):
              with argscope(Conv2D, padding = 'SAME'):
                b1 = l if first else BNReLU(l)
                c1 = Conv2D('conv1', b1, out_channel,
                            strides=stride1, activation=BNReLU)
                c2 = Conv2D('conv2', c1, out_channel)
                if increase_dim:
                    l = AvgPooling('pool', l, 2)
                    l = tf.pad(
                        l, [[0, 0], [in_channel//2, in_channel//2], [0, 0], [0, 0]])
                l = c2 + l
                return l
        with argscope([Conv2D, AvgPooling, BatchNorm, GlobalAvgPooling], data_format='channels_first'), \
             argscope(Conv2D, use_bias=False, kernel_size=3, kernel_initializer=tf.compat.v1.variance_scaling_initializer(scale=2.0, mode='fan_out')):
                l = Conv2D('conv0', image, 16, activation=nonlin.BNReLU, padding = 'SAME')
                l = residual('res1.0', l, first=True)
                for k in range(1, self.n):
                    l = residual('res1.{}'.format(k), l)
                # 32, c = 16
                l = residual('res2.0', l, increase_dim=True)
                for k in range(1, self.n):
                    l = residual('res2.{}'.format(k), l)
                # 16, c = 32
                l = residual('res3.0', l, increase_dim=True)
                for k in range(1, self.n):
                    l = residual('res3.{}'.format(k), l)
                l = nonlin.BNReLU('bnlast', l)
                # 8, c = 64
                l = GlobalAvgPooling('gap', l)
                
        logits = FullyConnected('linear', l, 10)
        tf.nn.softmax(logits, name='output')
        
        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=logits)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')
        wrong = tf.cast(tf.logical_not(tf.nn.in_top_k(label, logits, 1)), tf.float32,  name='wrong_vector')
        
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
        return tf.compat.v1.train.MomentumOptimizer(lr, 0.9)


def get_data(train_or_test):
    isTrain = train_or_test == 'train'
    ds = dataset.Cifar10(train_or_test)
    pp_mean = ds.get_per_pixel_mean()
    if isTrain:
        augumentors = [
            imgaug.CenterPaste((40, 40)),
            imgaug.RandomCrop((32, 32)),
            imgaug.Flip(horiz=True),
            imgaug.MapImage(lambda x: x - pp_mean)
        ]
    else:
        augumentors = [
            imgaug.MapImage(lambda x: x - pp_mean)
        ]
    ds = AugmentImageComponent(ds, augumentors)
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
