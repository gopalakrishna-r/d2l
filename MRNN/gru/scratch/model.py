import tensorflow as tf


def init_gru_state(batch_size, num_hiddens):
    return tf.zeros((batch_size, num_hiddens)),


def gru(inputs, state, params):
    Wxz, Whz, bz, Wxr, Whr, br, Wxh, Whh, bh, Whq, bq = params
    H, = state
    outputs = []

    for X in inputs:
        # reset gate
        X = tf.reshape(X, [-1, Wxh.shape[0]])
        R = tf.sigmoid(tf.matmul(X, Wxr) + tf.matmul(H, Whr) + br)
        Z = tf.sigmoid(tf.matmul(X, Wxz) + tf.matmul(H, Whz) + bz)

        # candidate hidden state
        H_tilde = tf.tanh(tf.matmul(X, Wxh) + tf.matmul((R * H), Whh) + bh)

        # hidden state
        H = (Z * H) + ((1 - Z) * H_tilde)

        # output
        outputs.append(tf.matmul(H, Whq) + bq)

    return tf.concat(outputs, axis=0), (H,)
