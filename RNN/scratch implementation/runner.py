import matplotlib.pyplot as plt
import tensorflow as tf
from d2l import tensorflow as d2l

from model import RNNModel, init_rnn_state, rnn
from parameters import get_params
from predictor import predict_ch8
from trainer import train_ch8

num_hiddens = 512
batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

train_random_iter, vocab_random_iter = d2l.load_data_time_machine(batch_size, num_steps, use_random_iter=True)

X = tf.reshape(tf.range(10), (2, 5))

net = RNNModel(len(vocab), num_hiddens, init_rnn_state, rnn, get_params)

state = net.begin_state(X.shape[0])
Y, new_state = net(X, state)
print(Y.shape, len(new_state), new_state[0].shape)

print(predict_ch8('time traveller ', 10, net, vocab))

device_name = d2l.try_gpu()._device_name
strategy = tf.distribute.OneDeviceStrategy(device_name)
num_epochs, lr = 500, 1
train_ch8(net, train_iter, vocab, lr, num_epochs, strategy)
plt.show()
