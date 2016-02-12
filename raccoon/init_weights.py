import numpy as np

from lasagne.init import Initializer


class SumZero(Initializer):
    """
    """
    def __init__(self, initializer):
        self.initializer = initializer

    def sample(self, shape):
        init_params = self.initializer.sample(shape)
        init_params -= init_params.mean(axis=-1, keepdims=True)
        return init_params
