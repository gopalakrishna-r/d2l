import matplotlib.pyplot as plt
import tensorflow as tf
from d2l import tensorflow as d2l
from model.LSTMCell import LSTMCell

from model.StackedRNNCell import StackedRNNCells

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
num_inputs = vocab_size
device_name = d2l.try_gpu()._device_name
strategy = tf.distribute.OneDeviceStrategy(device_name)
rnn_cells = [LSTMCell(num_hiddens) for _ in range(num_layers)]
stacked_lstm = StackedRNNCells(rnn_cells)
lstm_layer = tf.keras.layers.RNN(stacked_lstm, time_major=True, return_state=True,
                                 return_sequences=True)

with strategy.scope():
    model = d2l.RNNModel(lstm_layer, len(vocab))

num_epochs, lr = 500, 2
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, strategy)
plt.show()
