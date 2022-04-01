import tensorboard
import tensorflow as tf
from d2l import tensorflow as d2l
from pathlib import Path

from Inception import build_graph

net = build_graph

lr, num_epochs, batch_size = 0.1, 10, 32
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=lr,
    decay_steps=10000,
    decay_rate=0.9)
optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)

d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
