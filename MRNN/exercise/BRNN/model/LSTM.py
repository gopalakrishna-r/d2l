from typing import Tuple, Optional, overload
import tensorflow as tf
from tensorflow import Tensor, nest

import RNNBase


def apply_permutation(tensor: Tensor, permutation: Tensor, dim: int = 1) -> Tensor:
    return tf.gather(params=tensor, axis=dim, indices=permutation)


def permute_hidden(  # type: ignore[override]
        hx: Tuple[Tensor, Tensor],
        permutation: Optional[Tensor]
) -> Tuple[Tensor, Tensor]:
    if permutation is None:
        return hx
    return apply_permutation(hx[0], permutation), apply_permutation(hx[1], permutation)


class LSTM(RNNBase):

    def __init__(self, *args, **kwargs):
        super(LSTM, self).__init__('LSTM', *args, **kwargs)

    def get_expected_cell_size(self, input_cells: Tensor, batch_sizes: Optional[Tensor]) -> Tuple[int, int, int]:
        if batch_sizes is not None:
            mini_batch = int(batch_sizes[0])
        else:
            mini_batch = input_cells.shape(0) if self.batch_first else input_cells.size(1)
        num_directions = 2 if self.bidirectional else 1
        expected_hidden_size = (self.num_layers * num_directions,
                                mini_batch, self.hidden_size)
        return expected_hidden_size

    # In the future, we should prevent mypy from applying contravariance rules here.
    # See torch/nn/modules/module.py::_forward_unimplemented
    def check_forward_args(self,  # type: ignore[override]
                           input: Tensor,
                           hidden: Tuple[Tensor, Tensor],
                           batch_sizes: Optional[Tensor],
                           ):
        self.check_input(input, batch_sizes)
        self.check_hidden_size(hidden[0], self.get_expected_hidden_size(input, batch_sizes),
                               'Expected hidden[0] size {}, got {}')
        self.check_hidden_size(hidden[1], self.get_expected_cell_size(input, batch_sizes),
                               'Expected hidden[1] size {}, got {}')

    # Same as above, see torch/nn/modules/module.py::_forward_unimplemented

    # Same as above, see torch/nn/modules/module.py::_forward_unimplemented
    @overload  # type: ignore[override]
    def forward(self, input: Tensor, hx: Optional[Tuple[Tensor, Tensor]] = None
                ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:  # noqa: F811
        pass

    # Same as above, see torch/nn/modules/module.py::_forward_unimplemented
    @overload
    def forward(self, input, hx: Optional[Tuple[Tensor, Tensor]] = None
                ):  # noqa: F811
        pass

    def forward(self, input, hx=None):  # noqa: F811
        orig_input = input

        batch_sizes = None
        is_batched = input.dim() == 3
        batch_dim = 0 if self.batch_first else 1
        if not is_batched:
            input = tf.expand_dims(input,batch_dim)
        max_batch_size = input.size(0) if self.batch_first else input.size(1)
        sorted_indices = None
        unsorted_indices = None

        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            real_hidden_size = self.proj_size if self.proj_size > 0 else self.hidden_size
            h_zeros = tf.reshape((), self.num_layers * num_directions,
                                 max_batch_size, real_hidden_size)
            c_zeros = tf.reshape((), (self.num_layers * num_directions,
                                      max_batch_size, self.hidden_size))
            hx = (h_zeros, c_zeros)
        else:
            if is_batched:
                if (hx[0].dim() != 3 or hx[1].dim() != 3):
                    msg = ("For batched 3-D input, hx and cx should "
                           f"also be 3-D but got ({hx[0].dim()}-D, {hx[1].dim()}-D) tensors")
                    raise RuntimeError(msg)
            else:
                if hx[0].dim() != 2 or hx[1].dim() != 2:
                    msg = ("For unbatched 2-D input, hx and cx should "
                           f"also be 2-D but got ({hx[0].dim()}-D, {hx[1].dim()}-D) tensors")
                    raise RuntimeError(msg)
                hx = (tf.expand_dims(hx[0], 1), tf.expand_dims(hx[1], 1))

            # Each batch of the hidden state should match the input sequence that
            # the user believes he/she is passing in.
            hx = permute_hidden(hx, sorted_indices)

        self.check_forward_args(input, hx, batch_sizes)
        if batch_sizes is None:
            result = _VF.lstm(input, hx, self._flat_weights, self.bias, self.num_layers,
                              self.dropout, self.training, self.bidirectional, self.batch_first)
        else:
            result = _VF.lstm(input, batch_sizes, hx, self._flat_weights, self.bias,
                              self.num_layers, self.dropout, self.training, self.bidirectional)
        output = result[0]
        hidden = result[1:]
        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        if isinstance(orig_input, PackedSequence):
            output_packed = PackedSequence(output, batch_sizes, sorted_indices, unsorted_indices)
            return output_packed, permute_hidden(hidden, unsorted_indices)
        else:
            if not is_batched:
                output = output.squeeze(batch_dim)
                hidden = (hidden[0].squeeze(1), hidden[1].squeeze(1))
            return output, permute_hidden(hidden, unsorted_indices)

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