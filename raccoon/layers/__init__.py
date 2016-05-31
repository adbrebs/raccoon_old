import theano
from theano import shared
import theano.tensor as T

import numpy as np

from utils import create_uneven_weight

theano.config.floatX = 'float32'
floatX = theano.config.floatX


def normal_mat(size):
    return np.random.normal(0, 0.001, size=size).astype(floatX)


def create_parameter(initializer, shape, name=None):
    if isinstance(initializer, theano.tensor.sharedvar.SharedVariable):
        return initializer
    elif isinstance(initializer, np.ndarray):
        return shared(initializer)
    elif np.isscalar(initializer):
        return shared(initializer*np.ones(shape, dtype=floatX))

    return shared(initializer.sample(shape), name=name)

