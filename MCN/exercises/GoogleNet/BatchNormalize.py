import tensorflow as tf
from d2l import tensorflow as d2l


def batch_norm(bridge_input, gamma, beta, moving_mean, moving_var, eps):
    inv = tf.cast(tf.math.rsqrt(moving_var + eps), dtype=bridge_input.dtype)
    inv *= gamma
    return bridge_input * inv + (beta - moving_mean * inv)


def assign_moving_average(variable, value):
    momentum = 0.9
    delta = variable * momentum + value * (1 - momentum)
    return variable.assign(delta)


class BatchNorm(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(BatchNorm, self).__init__(**kwargs)
        self.gamma = None
        self.beta = None
        self.moving_mean = None
        self.moving_variance = None

    def build(self, input_shape):
        weight_shape = [input_shape[-1], ]
        self.gamma = self.add_weight(name="gamma", shape=weight_shape,
                                     initializer=tf.initializers.ones, trainable=True)
        self.beta = self.add_weight(name="beta", shape=weight_shape,
                                    initializer=tf.initializers.zeros, trainable=True)
        self.moving_mean = self.add_weight(name='moving_mean', shape=weight_shape,
                                           initializer=tf.initializers.zeros, trainable=False)
        self.moving_variance = self.add_weight(name='moving_variance', shape=weight_shape,
                                               initializer=tf.initializers.ones, trainable=False)
        super(BatchNorm, self).__init__(input_shape)

    @tf.function
    def call(self, inputs, training):
        if training:
            axes = list(range(len(inputs.shape) - 1))
            batch_mean = tf.reduce_mean(inputs, axes, keepdims=True)
            batch_variance = tf.reduce_mean(tf.math.squared_difference(inputs,tf.stop_gradient(batch_mean)),
                                            axes, keepdims=True)
            batch_mean = tf.squeeze(batch_mean, axes)
            batch_variance = tf.squeeze(batch_variance, axes)
            mean_update = assign_moving_average(self.moving_mean, batch_mean)
            variance_update = assign_moving_average(self.moving_variance, batch_variance)
            self.add_update(mean_update)
            self.add_update(variance_update)
            mean, variance = batch_mean, batch_variance
        else:
            mean, variance = self.moving_mean, self.moving_variance
        output = batch_norm(inputs, moving_mean=mean, moving_var=variance,
                            beta=self.beta, gamma=self.gamma, eps=1e-5)
        return output
