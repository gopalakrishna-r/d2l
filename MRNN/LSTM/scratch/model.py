import tensorflow as tf


def init_lstm_state(num_hiddens, batch_size):
    return (tf.zeros(shape=(batch_size, num_hiddens)),
            tf.zeros(shape=(batch_size, num_hiddens)))


def lstm(inputs, state, parameters):
    wxf, whf, bf, wxi, whi, bi, wxo, who, bo, wxc, whc, bc, whq, bq = parameters

    H, C = state
    outputs = []
    for X in inputs:
        X = tf.reshape(X, [-1, wxi.shape[0]])
        input_gate = tf.sigmoid(tf.matmul(X, wxi) + tf.matmul(H, whi) + bi)
        forget_gate = tf.sigmoid(tf.matmul(X, wxf) + tf.matmul(H, whf) + bf)
        output_gate = tf.sigmoid(tf.matmul(X, wxo) + tf.matmul(H, who) + bo)

        cand_tilda = tf.tanh(tf.matmul(X, wxc) + tf.matmul(H, whc) + bc)

        C = input_gate * cand_tilda + forget_gate * C
        H = output_gate * tf.tanh(C)

        outputs.append(tf.matmul(H, whq) + bq)

    return tf.concat(outputs, axis=0), (H, C)
