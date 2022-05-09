import tensorflow as tf


def init_rnn_state(batch_size, num_hiddens):
    return (tf.zeros((batch_size, num_hiddens)),)


def rnn(inputs, state, params):
    w_xh, w_hh, b_h, w_hq, b_q = params
    H, = state
    outputs = []

    for X in inputs:
        X = tf.reshape(X, [-1, w_xh.shape[0]])
        H = tf.tanh(tf.matmul(X, w_xh) + tf.matmul(H, w_hh) + b_h)
        Y = tf.matmul(H, w_hq) + b_q
        outputs.append(Y)
    return tf.concat(outputs, axis=0), (H,)


class RNNModel:
    def __init__(self, vocab_size, num_hiddens,
                 init_state, forward_fn, get_params):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.init_state, self.forward_fn = init_state, forward_fn
        self.trainable_variables = get_params(vocab_size, num_hiddens)

    def __call__(self, inputs, state):
        inputs = tf.one_hot(tf.transpose(inputs), self.vocab_size)
        inputs = tf.cast(inputs, tf.float32)
        return self.forward_fn(inputs, state, self.trainable_variables)

    def begin_state(self, batch_size):
        return self.init_state(batch_size, self.num_hiddens)
