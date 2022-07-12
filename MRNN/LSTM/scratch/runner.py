import tensorflow as tf
from d2l import tensorflow as d2l

from model import lstm, init_lstm_state
from parameters import get_lstm_parameters

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

vocab_size, num_hiddens, device_name = len(vocab), 256, d2l.try_gpu()._device_name
num_epochs, lr = 500, 1

strategy = tf.distribute.OneDeviceStrategy(device_name)
with strategy.scope():
    model = d2l.RNNModelScratch(len(vocab), num_hiddens, init_lstm_state, lstm, get_lstm_parameters)

d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, strategy)
