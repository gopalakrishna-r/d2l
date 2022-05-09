import tensorflow as tf
import tensorflow.keras as keras


class RNNOutputLayer(keras.layers.Layer):
    def __init__(self, rnn, vocab_size, **kwargs):
        super(RNNOutputLayer, self).__init__(**kwargs)
        self.rnn_layer = rnn
        self.vocab_size = vocab_size
        self.dense = keras.layers.Dense(vocab_size)

    def call(self, inputs, state):
        X = tf.one_hot(tf.transpose(inputs), self.vocab_size)
        Y, *state = self.rnn_layer(X, state)
        output = self.dense(tf.reshape(Y, [-1, Y.shape[-1]]))
        return output, state

    def begin_state(self, *args, **kwargs):
        return self.rnn_layer.cell.get_initial_state(*args, **kwargs)


