import tensorflow as tf
from d2l import tensorflow as d2l
import matplotlib.pyplot as plt
batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

vocab_size, num_hiddens, device_name = len(vocab), 256, d2l.try_gpu()._device_name
num_epochs, lr = 500, 1

lstm_cell = tf.keras.layers.LSTMCell(num_hiddens, kernel_initializer="glorot_uniform")
lstm_layer = tf.keras.layers.RNN(lstm_cell, time_major=True, return_sequences=True, return_state=True)

strategy = tf.distribute.OneDeviceStrategy(device_name)
with strategy.scope():
    model = d2l.RNNModel(lstm_layer, len(vocab))

d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, strategy)
plt.show()