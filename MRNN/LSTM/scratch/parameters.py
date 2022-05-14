import tf_slim as slim
import tensorflow as tf


def get_lstm_parameters(num_hiddens, vocab_size):
    with slim.arg_scope([slim.model_variable],
                        initializer=tf.random_normal_initializer(stddev=0.01, mean=0),
                        dtype=tf.float32, device=f'/GPU:0'):
        num_inputs = num_outputs = vocab_size
        # input weights
        with slim.arg_scope([slim.model_variable], shape=[num_inputs, num_hiddens]):
            wxi = slim.model_variable(name="wxi")  # input gate
            wxf = slim.model_variable(name="wxf")  # forget gate
            wxo = slim.model_variable(name="wxo")  # output gate
            wxc = slim.model_variable(name="wxc")  # candidate memory gate

        # hidden state weights
        with slim.arg_scope([slim.model_variable], shape=[num_hiddens, num_hiddens]):
            whi = slim.model_variable(name="whi")  # input gate
            whf = slim.model_variable(name="whf")  # forget gate
            who = slim.model_variable(name="who")  # output gate
            whc = slim.model_variable(name="whc")  # candidate memory gate

        # biases
        with slim.arg_scope([slim.model_variable], shape=[num_hiddens], initializer=tf.zeros_initializer()):
            bi = slim.model_variable(name="bi")  # input gate bias
            bf = slim.model_variable(name="bf")  # forget gate bias
            bo = slim.model_variable(name="bo")  # output gate bias
            bc = slim.model_variable(name="bc")  # candidate memory gate bias

        whq = slim.model_variable(name="whq", shape=[num_hiddens, num_outputs], dtype=tf.float32)
        bq = slim.model_variable(name="bq", shape=[num_outputs], initializer=tf.zeros_initializer(), dtype=tf.float32)

        params = [wxf, whf, bf, wxi, whi, bi, wxo, who, bo, wxc, whc, bc, whq, bq]
        return params
