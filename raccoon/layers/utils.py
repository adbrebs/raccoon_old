import numpy as np
from theano import tensor as T


def to_list(input):
    # make sure that the input is a list
    if not isinstance(input, (tuple, list)):
        input = [input]
    return input


def clip_norm_gradients(grads, value=1):
    n = T.sqrt(sum([T.sum(T.square(g)) for g in grads]))
    return [T.switch(n >= value, g * value / n, g) for g in grads]


def create_uneven_weight(ls_n_in, n_out, initializer):
    # This allows to have different initial scales for different parts of
    # the input.
    ls_w_in_mat = []
    for n_in in ls_n_in:
        ls_w_in_mat.append(initializer.sample((n_in, n_out)))
    return np.concatenate(ls_w_in_mat, axis=0) / len(ls_n_in)


def logsumexp(x, axis=None):
    """
    Efficient log of a sum of exponentials
    """
    x_max = T.max(x, axis=axis, keepdims=True)
    z = T.log(T.sum(T.exp(x - x_max), axis=axis, keepdims=True)) + x_max
    return z.sum(axis=axis)