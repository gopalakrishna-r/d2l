import matplotlib.pyplot as plt
import tensorflow as tf
from d2l import tensorflow as d2l
from d2l.tensorflow import train_ch8

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

vocab_size, num_hiddens, device_name = len(vocab), 256, d2l.try_gpu()._device_name
strategy = tf.distribute.OneDeviceStrategy(device_name)

gru_cell = tf.keras.layers.GRUCell(num_hiddens, kernel_initializer='glorot_uniform')
gru_layer = tf.keras.layers.RNN(gru_cell, time_major=True, return_state=True, return_sequences=True)

num_epochs, lr = 500, 1
with strategy.scope():
    model = d2l.RNNModel(gru_layer, vocab_size=len(vocab))

train_ch8(model, train_iter, vocab, lr, num_epochs, strategy)
plt.show()
