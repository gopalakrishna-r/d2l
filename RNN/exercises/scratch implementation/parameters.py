import tf_slim as slim
import tensorflow as tf


# initializing the model parameters

def get_params(vocab_size, num_hiddens):
    num_inputs = num_outputs = vocab_size
    with slim.arg_scope([slim.model_variable],
                        initializer=tf.random_normal_initializer(stddev=0.01, mean=0),
                        dtype=tf.float32, device=f'/GPU:0'):
        # hidden layer parameters
        w_xh = slim.model_variable('w_xh', shape=(num_inputs, num_hiddens))
        w_hh = slim.model_variable('w_hh', shape=(num_hiddens, num_hiddens))
        b_h = slim.model_variable('b_h', shape=num_hiddens, initializer=tf.zeros_initializer(), dtype=tf.float32)
        # output layer parameters
        w_hq = slim.model_variable('w_hq', shape=(num_hiddens, num_outputs))
        b_q = slim.model_variable('b_q', shape=num_outputs, initializer=tf.zeros_initializer(), dtype=tf.float32)

        params = [w_xh, w_hh, b_h, w_hq, b_q]
        return params
