import builtins

from tensorflow import Tensor


class Parameter(Tensor):
    def __init__(self, data: Tensor = ..., requires_grad: builtins.bool = ...): ...

    ...
