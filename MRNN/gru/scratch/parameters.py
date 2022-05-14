import tensorflow as tf
import tf_slim as slim


def get_params(vocab_size, num_hiddens):
    with slim.arg_scope([slim.model_variable],
                        initializer=tf.random_normal_initializer(stddev=0.01, mean=0),
                        dtype=tf.float32, device=f'/GPU:0'):
        num_inputs = num_outputs = vocab_size
        # input weights
        with slim.arg_scope([slim.model_variable], shape=[num_inputs, num_hiddens]):
            Wxr = slim.model_variable(name="Wxr")  # reset gate
            Wxz = slim.model_variable(name="Wxz")  # update gate
            Wxh = slim.model_variable(name="Wxh")  # candidate hidden gate

        # hidden state weights
        with slim.arg_scope([slim.model_variable], shape=[num_hiddens, num_hiddens]):
            Whr = slim.model_variable(name="Whr")  # reset gate
            Whz = slim.model_variable(name="Whz")  # update gate
            Whh = slim.model_variable(name="Whh")  # candidate hidden gate

        # biases
        with slim.arg_scope([slim.model_variable], shape=[num_hiddens], initializer=tf.zeros_initializer()):
            br = slim.model_variable(name="br")
            bz = slim.model_variable(name="bz")
            bh = slim.model_variable(name="bh")

        Whq = slim.model_variable(name="Whq", shape=[num_hiddens, num_outputs], dtype=tf.float32)
        bq = slim.model_variable(name="bq", shape=[num_outputs], initializer=tf.zeros_initializer(), dtype=tf.float32)

        params = [Wxz, Whz, bz, Wxr, Whr, br, Wxh, Whh, bh, Whq, bq]
        return params
