import theano
import theano.tensor as T

import lasagne

floatX = theano.config.floatX

def poly_softmax(eps=1e-2, order=2):
    def f(x):
        x = x**order + eps
        return x / T.sum(x, axis=1, keepdims=True)
    return f


def taylor_softmax(order=2):
    def f(x):
        y = 1.
        fa = 1.
        for i in range(1, order+1):
            fa *= i
            y += x**i / fa
        return y / y.sum(axis=1, keepdims=True)
    return f


def double_act(p, act1, act2):
    def f(x):
        return p*act1(x) + (1-p)*act2(x)
    return f



def taylor_softmax_bis(order=2):
    def f(x):
        y = 1.
        fa = 1.
        x = x - x.mean(axis=1, keepdims=True)
        for i in range(1, order+1):
            fa *= i
            y += x**i / fa
        return y / y.sum(axis=1, keepdims=True)
    return f



def softplus_softmax():
    def f(x):
        x = T.nnet.softplus(x)
        return x / T.sum(x, axis=1, keepdims=True)
    return f


def relu_poly(order=2):
    def f(x):
        return T.switch(x < 0, 0, x**order)
    return f


def softmax_abs(x):
    return lasagne.nonlinearities.softmax(T.abs_(x))


def compute_rank_metrics(y_hat, y, mask=None, ks=10):
    """
    Computes metrics related to the rank of the prediction:
    - mean rank
    - mean inverse rank
    - top-k for different values of k. ks can either be a list or a scalar.

    mask and y_hat should have the same first dimension
    """
    if not isinstance(ks, list):
        ks = [ks]

    rank = T.sum(y_hat >= y_hat[T.arange(y_hat.shape[0]), y][:, None], axis=1)

    if mask:
        n_not_masked = mask.sum()
    def mean(tensor):
        if mask:
            return (tensor * mask[:, None]).sum() / T.cast(n_not_masked, floatX)
        else:
            return tensor.sum() / T.cast(tensor.size, floatX)

    topks = []
    for k in ks:
        res = mean(rank > k)
        res.name = "k={} error rate".format(k)
        topks.append(res)

    mean_rank = mean(rank)
    std_rank = rank.std()
    mean_rank.name = 'mean_rank'
    mean_inv_rank = mean(1.0/rank)
    mean_inv_rank.name = 'mean_inv_rank'
    mean_log_rank = mean(T.log(rank))
    mean_log_rank.name = 'mean_log_rank'
    return mean_rank, std_rank, mean_inv_rank, mean_log_rank, topks


def compute_rank_metrics_no_mean(y_hat, y, ks=10):
    """
    Computes metrics related to the rank of the prediction:
    - mean rank
    - mean inverse rank
    - top-k for different values of k. ks can either be a list or a scalar.
    """
    if not isinstance(ks, list):
        ks = [ks]

    rank = T.sum(y_hat >= y_hat[T.arange(y_hat.shape[0]), y][:, None], axis=1)

    topks = []
    for k in ks:
        res = rank > k
        res.name = "k={} error rate".format(k)
        topks.append(res)

    rank = rank
    rank.name = 'mean_rank'
    inv_rank = (1.0/rank)
    inv_rank.name = 'mean_inv_rank'
    log_rank = (T.log(rank))
    log_rank.name = 'mean_log_rank'
    return rank, inv_rank, log_rank, topks
