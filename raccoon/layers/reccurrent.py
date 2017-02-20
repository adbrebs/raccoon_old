import theano
from theano import shared
import theano.tensor as T
from theano.gradient import grad_clip

import numpy as np

from utils import create_uneven_weight

theano.config.floatX = 'float32'
floatX = theano.config.floatX


def normal_mat(size):
    return np.random.normal(0, 0.001, size=size).astype(floatX)


class GRULayer:
    """Classic GRU layer"""

    def __init__(self, ls_n_in, n_out, initializer, grad_clipping=None):
        """
        Parameters
        ----------
        ls_n_in: integer or list of integers
            Dimension of the inputs
            Allows to have different initializations for different parts of the
            input.
            If there is a single input, you can provide the single integer
            directly
        initializer: object with a sample(size) method
            Corresponding initializer of the weights

        grad_clipping: float
            Hard clip the gradients at each time step. Only the gradient values
            above this threshold are clipped by to the threshold. This is done
            during backprop.
        """
        if not isinstance(ls_n_in, (tuple, list)):
            ls_n_in = [ls_n_in]

        self.ls_n_in = ls_n_in
        self.n_in = sum(ls_n_in)
        self.n_out = n_out
        self.grad_clipping = grad_clipping

        # This allows to have different initial scales for different parts of
        # the input. For example when the input of the gru layer is the
        # concatenation of heteregoneous inputs.
        w_in_mat = create_uneven_weight(ls_n_in, 3 * n_out, initializer)

        self.w_in = shared(w_in_mat, 'w_in_gru')
        self.b_in = shared(normal_mat((3*n_out,)), 'b_in_gru')

        self.w_hid = shared(initializer.sample((n_out, 3 * n_out)),
                            'w_hid_gru')

        self.params = [self.w_in, self.b_in, self.w_hid]

    def precompute_inputs(self, inputs):
        return T.dot(inputs, self.w_in) + self.b_in

    def step(self, inputs, h_pre, mask=None, process_inputs=False):
        """
        A single timestep.

        Parameters
        ----------
        inputs: (batch_size, n_in)
        h_pre: (batch_size, n_hidden)
        mask: (batch_size,)

        process_inputs: bool
            If true, will process the input.
            If possible, it is better to process the whole input sequence
            beforehand. But sometimes this is not suitable, for example at
            prediction time.
        """
        n_out = h_pre.shape[1]

        if process_inputs:
            inputs = self.precompute_inputs(inputs)

        h_input = T.dot(h_pre, self.w_hid)

        sig = T.nnet.sigmoid

        gates = sig(inputs[:, :2*n_out] + h_input[:, :2*n_out])
        r_gate = gates[:, :n_out]
        u_gate = gates[:, n_out:2*n_out]
        h_new = T.tanh(inputs[:, 2*n_out:] + r_gate * h_input[:, 2*n_out:])

        h = (1-u_gate)*h_pre + u_gate*h_new

        if mask:
            h = T.switch(mask[:, None], h, h_pre)

        if self.grad_clipping:
            h = grad_clip(h, -self.grad_clipping, self.grad_clipping)

        return h

    def apply(self, seq_inputs, h_ini, seq_mask=None, go_backwards=False):
        """
        Recurse over the whole sequences

        Parameters
        ----------
        seq_inputs: (length_sequence, batch_size, n_in)
        seq_inputs: (length_sequence, batch_size)
        h_ini: (batch_size, n_hidden)
            Initial hidden state
        """

        seq_inputs = self.precompute_inputs(seq_inputs)

        def gru_step_no_mask(inputs, h_pre):
            return self.step(inputs, h_pre, mask=None, process_inputs=False)

        def gru_step_mask(inputs, mask, h_pre):
            return self.step(inputs, h_pre, mask=mask, process_inputs=False)

        if seq_mask:
            gru_step = gru_step_mask
            sequences = [seq_inputs, seq_mask]
        else:
            gru_step = gru_step_no_mask
            sequences = [seq_inputs]

        seq_h, scan_updates = theano.scan(
            fn=gru_step,
            sequences=sequences,
            outputs_info=[h_ini],
            go_backwards=go_backwards)

        return seq_h, scan_updates


class RnnCovarianceLayer:
    """
    A layer that takes a recurrent layer as input and compute the covariance
    matrix of its hidden states iteratively.
    Computing the covariance matrix at the end of training would be more
    expensive in memory.
    Note that backpropagating through this layer will keep all the intermediate
    covariance matrices and thus there is no memory gain.
    """
    def __init__(self, rnn_layer):
        self.rnn_layer = rnn_layer

    def step(self, inputs, h_pre, covariance_pre, mask=None,
             process_inputs=False):
        """
        Parameters
        ----------
        covariance_pre: (batch_size, n_hidden, n_hidden)
        """
        h = self.rnn_layer.step(inputs, h_pre, mask, process_inputs)

        inc_covariance = h.dimshuffle((0, 'x', 1)) * h.dimshuffle((0, 1, 'x'))

        return h, covariance_pre + inc_covariance

    def apply(self, seq_inputs, seq_mask, h_ini):
        """
        Parameters
        ----------
        covariance_ini: (batch_size, n_hidden, n_hidden)
        """
        seq_inputs = self.rnn_layer.precompute_inputs(seq_inputs)

        def rnn_step(inputs, mask, h_pre, covariance_pre):
            return self.step(inputs, h_pre, covariance_pre, mask,
                             process_inputs=False)

        covariance_ini = h_ini.dimshuffle((0, 'x', 1)) * h_ini.dimshuffle((0, 1, 'x'))
        (seq_h, seq_covariance), scan_updates = theano.scan(
            fn=rnn_step,
            sequences=[seq_inputs, seq_mask],
            outputs_info=[h_ini, covariance_ini])

        return (seq_h, seq_covariance[-1]), scan_updates


class BidirectionalRNN:
    def __init__(self, rnn_forward, rnn_backward):
        self.rnn_forward = rnn_forward
        self.rnn_backward = rnn_backward

        self.params = rnn_forward.params + rnn_backward.params

    def apply(self, seq_inputs, h_ini, seq_mask=None):
        fw_n_out = self.rnn_forward.n_out
        h_ini_forward = h_ini[:, :fw_n_out]
        h_ini_backward = h_ini[:, fw_n_out:]

        seq_forward, up_forward = self.rnn_forward.apply(
            seq_inputs, h_ini_forward, seq_mask=seq_mask)
        seq_backward, up_backward = self.rnn_backward.apply(
            seq_inputs, h_ini_backward, seq_mask=seq_mask, go_backwards=True)

        seq_forward *= seq_mask[:, :, None]
        seq_backward *= seq_mask[::-1, :, None]

        return (T.concatenate([seq_forward, seq_backward[::-1]], -1),
                up_forward + up_backward)


# class DeepRnn:
#     def __init__(self, ls_rnns):
#         self.ls_rnns = ls_rnns
#
#         self.params = [p for params in ls_rnns for p in params]
#
#     def apply(self, seq_inputs, h_ini, seq_mask=None):
#         fw_n_out = self.rnn_forward.n_out
#         h_ini_forward = h_ini[:, :fw_n_out]
#         h_ini_backward = h_ini[:, fw_n_out:]
#
#         seq_forward, up_forward = self.rnn_forward.apply(
#             seq_inputs, h_ini_forward, seq_mask=seq_mask)
#         seq_backward, up_backward = self.rnn_backward.apply(
#             seq_inputs, h_ini_backward, seq_mask=seq_mask, go_backwards=True)
#
#         seq_forward *= seq_mask[:, :, None]
#         seq_backward *= seq_mask[::-1, :, None]
#
#         return (T.concatenate([seq_forward, seq_backward[::-1]], -1),
#                 up_forward + up_backward)