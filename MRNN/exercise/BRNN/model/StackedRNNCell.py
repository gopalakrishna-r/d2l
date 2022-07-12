import functools

import tensorflow as tf
import tensorflow.keras as keras


class StackedRNNCells(keras.layers.Layer):

    def __init__(self, cells, **kwargs):

        self.cells = cells
        self.reverse_state_order = kwargs.pop('reverse_state_order', False)

        super(StackedRNNCells, self).__init__(**kwargs)

    @property
    def state_size(self):
        return tuple(c.state_size for c in
                     (self.cells[::-1] if self.reverse_state_order else self.cells))

    @property
    def output_size(self):

        def _is_multiple_state(state_size):
            """Check whether the state_size contains multiple states."""
            return (hasattr(state_size, '__len__') and
                    not isinstance(state_size, tf.TensorShape))

        if getattr(self.cells[-1], 'output_size', None) is not None:
            return self.cells[-1].output_size
        elif _is_multiple_state(self.cells[-1].state_size):
            return self.cells[-1].state_size[0]
        else:
            return self.cells[-1].state_size

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        initial_states = []
        for cell in self.cells[::-1] if self.reverse_state_order else self.cells:
            get_initial_state_fn = getattr(cell, 'get_initial_state', None)
            initial_states.append(get_initial_state_fn(
                inputs=inputs, batch_size=batch_size, dtype=dtype))

        return tuple(initial_states)

    def call(self, inputs, states, constants=None, training=None, **kwargs):
        # Recover per-cell states.
        state_size = (self.state_size[::-1]
                      if self.reverse_state_order else self.state_size)
        nested_states = tf.nest.pack_sequence_as(state_size, tf.nest.flatten(states))

        # Call the cells in order and store the returned states.
        new_nested_states = []
        for cell, states in zip(self.cells, nested_states):
            states = states if tf.nest.is_nested(states) else [states]
            # TF cell does not wrap the state into list when there is only one state.
            is_tf_rnn_cell = getattr(cell, '_is_tf_rnn_cell', None) is not None
            states = states[0] if len(states) == 1 and is_tf_rnn_cell else states

            # Use the __call__ function for callable objects, eg layers, so that it
            # will have the proper name scopes for the ops, etc.
            cell_call_fn = cell.__call__ if callable(cell) else cell.call
            inputs, states = cell_call_fn(inputs, states, **kwargs)
            new_nested_states.append(states)

        return inputs, tf.nest.pack_sequence_as(state_size,
                                                tf.nest.flatten(new_nested_states))

    def build(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        def get_batch_input_shape(batch_size, dim):
            shape = tf.TensorShape(dim).as_list()
            return tuple([batch_size] + shape)

        def _is_multiple_state(state_size):
            """Check whether the state_size contains multiple states."""
            return (hasattr(state_size, '__len__') and
                    not isinstance(state_size, tf.TensorShape))

        for cell in self.cells:
            if isinstance(cell, keras.layers.Layer) and not cell.built:
                with tf.name_scope(cell.name):
                    cell.build(input_shape)
                    cell.built = True
            if getattr(cell, 'output_size', None) is not None:
                output_dim = cell.output_size
            elif _is_multiple_state(cell.state_size):
                output_dim = cell.state_size[0]
            else:
                output_dim = cell.state_size
            batch_size = tf.nest.flatten(input_shape)[0]
            if tf.nest.is_nested(output_dim):
                input_shape = tf.nest.map_structure(
                    functools.partial(get_batch_input_shape, batch_size), output_dim)
                input_shape = tuple(input_shape)
            else:
                input_shape = tuple([batch_size] + tf.TensorShape(output_dim).as_list())
        self.built = True
