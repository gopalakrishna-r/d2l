import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from d2l import tensorflow as d2l

from model import RNNOutputLayer

batch_size = 32
num_hidden_state = 256
num_steps = 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

rnn_cell = keras.layers.SimpleRNNCell(num_hidden_state, kernel_initializer=keras.initializers.glorot_uniform)
rnn_layer = keras.layers.RNN(rnn_cell, time_major=True, return_sequences=True, return_state=True)

rnn_state = rnn_cell.get_initial_state(batch_size=batch_size, dtype=tf.float32)
print(rnn_state.shape)

X = tf.random.uniform((num_steps, batch_size, len(vocab)))
Y, state_new = rnn_layer(X, rnn_state)
print(Y.shape, len(state_new), state_new[0].shape)


device_name = d2l.try_gpu()._device_name
strategy = tf.distribute.OneDeviceStrategy(device_name)

with strategy.scope():
    net = RNNOutputLayer(rnn_layer, vocab_size=len(vocab))

print(d2l.predict_ch8('time traveller', 10, net, vocab))

num_epochs, lr = 500, 1
d2l.train_ch8(net, train_iter, vocab, lr, num_epochs, strategy)
plt.show()
