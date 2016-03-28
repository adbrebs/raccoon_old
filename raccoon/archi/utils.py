import numpy as np
from theano import tensor as T


def clip_norm_gradients(grads, value=5):
    n = T.sqrt(sum([T.sum(T.square(g)) for g in grads]))
    return [T.switch(n >= value, g * value / n, g) for g in grads]


def create_uneven_weight(ls_n_in, n_out, initializer):
    # This allows to have different initial scales for different parts of
    # the input.
    ls_w_in_mat = []
    for n_in in ls_n_in:
        ls_w_in_mat.append(initializer.sample((n_in, n_out)))
    return np.concatenate(ls_w_in_mat, axis=0) / len(ls_n_in)
