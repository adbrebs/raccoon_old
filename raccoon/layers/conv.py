import numpy as np

import theano
import theano.tensor as T
from theano import shared

from . import create_parameter


def floatX(x):
    """
    Convert `x` to the numpy type specified in `theano.config.floatX`.
    """
    if theano.config.floatX == 'float16':
        return np.float16(x)
    elif theano.config.floatX == 'float32':
        return np.float32(x)
    else:  # Theano's default float type is float64
        print "Warning: lib.floatX using float64"
    return np.float64(x)


def gated_non_linerity(x):
    gate = x[:, ::2]
    val = x[:, 1::2]
    return T.tanh(val) * T.nnet.sigmoid(gate)


def elu(x):
    return T.switch(x >= 0., x, T.exp(x) - floatX(1.))


def identity(x):
    return x


def highway(x, inp, channel_axis=2):
    assert (channel_axis in [1, 2])
    if channel_axis == 1:
        H_ = T.tanh(x[:, ::2])
        T_ = T.nnet.sigmoid(x[:, 1::2])
    else:
        H_ = T.tanh(x[:, :, ::2])
        T_ = T.nnet.sigmoid(x[:, :, 1::2])
    return H_*T_ + (1.-T_)*inp


class HighWayBlock:
    def __init__(
         self, conv_dim, K, filter_size, init, border_mode='pad_before'):
        self.conv_list = [
            Conv1d(
                   conv_dim, 2*conv_dim, filter_size, init=init,
                   non_linearity='identity', border_mode=border_mode
                ) for k in range(K)
        ]
        self.params = []
        for conv in self.conv_list:
            self.params.extend(conv.params)

    def apply(self, x, x_mask):
        output = x
        for i, conv in enumerate(self.conv_list):
            output = highway(conv.apply(output, x_mask), output)
        return output


class MultiFilterConv:
    def __init__(
         self, input_dim, output_dim_per_filter, K,
         init, border_mode='pad_before'):
        self.conv_list = [
            Conv1d(
                   input_dim, output_dim_per_filter,
                   k, init=init, border_mode=border_mode
                ) for k in range(1, K+1, 1)
        ]
        self.params = []
        for conv in self.conv_list:
            self.params.extend(conv.params)

    def apply(self, x, x_mask=None):
        output_list = [
            conv.apply(x, x_mask) for conv in self.conv_list
        ]

        return T.concatenate(output_list, axis=2)


class Conv1d:
    def __init__(
        self,
        input_dim,
        output_dim,
        filter_size,
        init=None,
        non_linearity='relu',
        bias=True,
        dim_order='tbd',
        border_mode='pad_before'
     ):

        assert(dim_order in ['tbd', 'btd'])
        assert(border_mode in ['pad_before', 'valid', 'half'])

        self.border_mode = border_mode
        self.dim_order = dim_order
        self.bias = bias
        self.non_linearity = non_linearity
        self.filter_size = filter_size

        if (filter_size == 1) and (self.border_mode != 'valid'):
            print "Warning: for filter-size = 1, only valid conv is supported"
            self.border_mode = 'valid'

        if self.border_mode == 'pad_before':
            self.border_mode_ = (self.filter_size - 1, 0)
        else:
            self.border_mode_ = self.border_mode

        if non_linearity in ['gated']:
            num_filters = 2*output_dim
        else:
            num_filters = output_dim

        W_shape = (num_filters, input_dim, filter_size, 1)

        if bias:
            bias_shape = (num_filters,)

        self.W = create_parameter(init, W_shape, "conv_filter")

        if bias:
            self.b = shared(
                floatX(np.random.normal(0, 0.01, size=bias_shape)), "conv_bias")
        self.params = [self.W, self.b]

        if non_linearity == 'gated':
            self.activation = gated_non_linerity
        elif non_linearity == 'relu':
            self.activation = T.nnet.relu
        elif non_linearity == 'elu':
            self.activation = elu
        elif non_linearity == 'identity':
            self.activation = identity
        else:
            raise NotImplementedError(
                "{} non-linearity not implemented!".format(non_linearity))

    def apply(self, x, x_mask=None):
        if self.dim_order == 'tbd':
            inp = x.dimshuffle(1, 2, 0, 'x')
        else:
            inp = x.dimshuffle(0, 2, 1, 'x')

        conv_out = T.nnet.conv2d(
                        inp,  self.W,
                        filter_flip=False,
                        border_mode=self.border_mode_
                    )

        if self.bias:
            conv_out = conv_out + self.b[None, :, None, None]

        output = self.activation(conv_out)

        if self.border_mode == 'half':
            if (self.filter_size % 2) == 0:
                output = output[:, :, :-1]
        elif self.border_mode == 'pad_before':
            output = output[:, :, :-self.filter_size+1]

        output = output.reshape(
            (output.shape[0], output.shape[1], output.shape[2]))

        if self.dim_order == 'tbd':
            output = output.dimshuffle(2, 0, 1)
        else:
            output = output.dimshuffle(0, 2, 1)

        if x_mask is not None:
            output = output*x_mask[:, :, None]

        return output
