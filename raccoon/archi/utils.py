from theano import tensor as T


def clip_norm_gradients(grads, value=5):
    n = T.sqrt(sum([T.sum(T.square(g)) for g in grads]))
    return [T.switch(n >= value, g * value / n, g) for g in grads]
