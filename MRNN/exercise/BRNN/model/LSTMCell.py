import tensorflow as tf
import tensorflow.keras as keras


class LSTMCell(keras.layers.Layer):

    def __init__(self,
                 units,
                 activation='tanh',
                 use_bias=True,
                 recurrent_activation='hard_sigmoid',
                 **kwargs):
        if units < 0:
            raise ValueError(f'Received an invalid value for argument `units`, '
                             f'expected a positive integer, got {units}.')
        # By default use cached variable under v2 mode, see b/143699808.

        super(LSTMCell, self).__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)

        self.recurrent_activation = keras.activations.get(recurrent_activation)
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.RandomNormal(stddev=0.01, mean=0)
        self.bias_initializer = keras.initializers.zeros()
        implementation = kwargs.pop('implementation', 1)
        self.implementation = implementation
        self.state_size = [self.units, self.units]
        self.output_size = self.units

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.kernel = tf.Variable(self.kernel_initializer(shape=(input_dim, self.units * 4)),
                                  trainable=True)
        self.recurrent_kernel = tf.Variable(
            self.kernel_initializer(shape=(self.units, self.units * 4)),
            trainable=True)

        if self.use_bias:
            self.bias = tf.Variable(self.bias_initializer(shape=(self.units * 4)), trainable=True)
        else:
            self.bias = None

        self.built = True

    def _compute_carry_and_output(self, x, h_tm1, c_tm1):
        """Computes carry and output using split kernels."""
        x_i, x_f, x_c, x_o = x
        h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o = h_tm1
        i = self.recurrent_activation(
            x_i + tf.matmul(h_tm1_i, self.recurrent_kernel[:, :self.units]))
        f = self.recurrent_activation(x_f + tf.matmul(
            h_tm1_f, self.recurrent_kernel[:, self.units:self.units * 2]))
        c = f * c_tm1 + i * self.activation(x_c + tf.matmul(
            h_tm1_c, self.recurrent_kernel[:, self.units * 2:self.units * 3]))
        o = self.recurrent_activation(
            x_o + tf.matmul(h_tm1_o, self.recurrent_kernel[:, self.units * 3:]))
        return c, o

    def _compute_carry_and_output_fused(self, z, c_tm1):
        """Computes carry and output using fused kernels."""
        z0, z1, z2, z3 = z
        i = self.recurrent_activation(z0)
        f = self.recurrent_activation(z1)
        c = f * c_tm1 + i * self.activation(z2)
        o = self.recurrent_activation(z3)
        return c, o

    def call(self, inputs, states, training=None):
        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state

        if self.implementation == 1:
            inputs_i = inputs
            inputs_f = inputs
            inputs_c = inputs
            inputs_o = inputs
            k_i, k_f, k_c, k_o = tf.split(
                self.kernel, num_or_size_splits=4, axis=1)
            x_i = tf.matmul(inputs_i, k_i)
            x_f = tf.matmul(inputs_f, k_f)
            x_c = tf.matmul(inputs_c, k_c)
            x_o = tf.matmul(inputs_o, k_o)
            if self.use_bias:
                b_i, b_f, b_c, b_o = tf.split(
                    self.bias, num_or_size_splits=4, axis=0)
                x_i = bias_add(x_i, b_i)
                x_f = bias_add(x_f, b_f)
                x_c = bias_add(x_c, b_c)
                x_o = bias_add(x_o, b_o)

            h_tm1_i = h_tm1
            h_tm1_f = h_tm1
            h_tm1_c = h_tm1
            h_tm1_o = h_tm1
            x = (x_i, x_f, x_c, x_o)
            h_tm1 = (h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o)
            c, o = self._compute_carry_and_output(x, h_tm1, c_tm1)
        else:
            z = tf.matmul(inputs, self.kernel)
            z += tf.matmul(h_tm1, self.recurrent_kernel)
            if self.use_bias:
                z = bias_add(z, self.bias)

            z = tf.split(z, num_or_size_splits=4, axis=1)
            c, o = self._compute_carry_and_output_fused(z, c_tm1)

        h = o * self.activation(c)
        return h, [h, c]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return list(_generate_zero_filled_state_for_cell(
            self, inputs, batch_size, dtype))


def bias_add(x, bias, data_format='NHWC'):
    return tf.nn.bias_add(x, bias, data_format=data_format)


def _generate_zero_filled_state_for_cell(cell, inputs, batch_size, dtype):
    if inputs is not None:
        batch_size = tf.shape(inputs)[0]
        dtype = inputs.dtype
    return _generate_zero_filled_state(batch_size, cell.state_size, dtype)


def _generate_zero_filled_state(batch_size_tensor, state_size, dtype):
    """Generate a zero filled tensor with shape [batch_size, state_size]."""
    if batch_size_tensor is None or dtype is None:
        raise ValueError(
            'batch_size and dtype cannot be None while constructing initial state. '
            f'Received: batch_size={batch_size_tensor}, dtype={dtype}')

    def create_zeros(unnested_state_size):
        flat_dims = tf.TensorShape(unnested_state_size).as_list()
        init_state_size = [batch_size_tensor] + flat_dims
        return tf.zeros(init_state_size, dtype=dtype)

    if tf.nest.is_nested(state_size):
        return tf.nest.map_structure(create_zeros, state_size)
    else:
        return create_zeros(state_size)
