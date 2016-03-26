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

    def __init__(self, ls_n_in, n_out, ls_initializers, grad_clipping=100):
        """
        Parameters
        ----------
        ls_n_in: integer or list of integers
            Dimension of the inputs
            Allows to have different initializations for different parts of the
            input.
            If there is a single input, you can provide the single integer
            directly
        ls_initializers: object with a sample(size) method
            Corresponding initializers

        grad_clipping: float
            Hard clip the gradients at each time step. Only the gradient values
            above this threshold are clipped by to the threshold. This is done
            during backprop.
        """
        if len(ls_n_in) != len(ls_initializers):
            raise ValueError('ls_n_in should have the same length as '
                             'ls_initializers')

        if not isinstance(ls_n_in, (tuple, list)):
            ls_n_in = [ls_n_in]
        if not isinstance(ls_initializers, (tuple, list)):
            ls_initializers = [ls_initializers]

        self.ls_n_in = ls_n_in
        self.n_in = sum(ls_n_in)
        self.n_out = n_out
        self.grad_clipping = grad_clipping

        # This allows to have different initial scales for different parts of
        # the input. For example when the input of the gru layer is the
        # concatenation of heteregoneous inputs.
        w_in_mat = create_uneven_weight(ls_n_in, 3 * n_out, ls_initializers)

        self.w_in = shared(w_in_mat, 'w_in_gru')
        self.b_in = shared(normal_mat((3*n_out,)), 'b_in_gru')

        self.w_rec = shared(ls_initializers.sample((n_out, n_out)), 'w_rec_gru')

        self.w_gates = shared(ls_initializers.sample((n_out, 2 * n_out)),
                              'w_gates_gru')

        self.params = [self.w_in, self.b_in, self.w_rec, self.w_gates]

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

        x_in = inputs[:, :n_out]
        x_gates = inputs[:, n_out:]

        gates = T.nnet.sigmoid(x_gates + T.dot(h_pre, self.w_gates))
        r_gate = gates[:, :n_out]
        u_gate = gates[:, n_out:]

        h_new = T.tanh(x_in + T.dot(r_gate * h_pre, self.w_rec))

        h = (1-u_gate)*h_pre + u_gate*h_new

        if mask:
            h = mask[:, None]*h + (1-mask[:, None])*h_pre

        h = grad_clip(h, -self.grad_clipping, self.grad_clipping)

        return h

    def apply(self, seq_inputs, seq_mask, h_ini):
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

        def gru_step(inputs, mask, h_pre):
            return self.step(inputs, h_pre, mask, process_inputs=False)

        seq_h, scan_updates = theano.scan(
            fn=gru_step,
            sequences=[seq_inputs, seq_mask],
            outputs_info=[h_ini])

        return seq_h, scan_updates


class PositionAttentionLayer:
    """
    Positional attention mechanism as described by Alex Graves in
    http://arxiv.org/abs/1308.0850
    """
    def __init__(self, layer_to_be_conditioned, n_in_cond, n_mixt, initializer):
        self.n_in_cond = n_in_cond
        self.n_mixt = n_mixt

        self.layer = layer_to_be_conditioned
        n_out = self.n_out = self.layer.n_out

        self.w_cond = shared(initializer.sample((n_out, 3*n_mixt)), 'w_cond')
        self.b_cond = shared(normal_mat((3*n_mixt, )), 'b_cond')

        self.params = layer_to_be_conditioned.params + \
                      [self.w_cond, self.b_cond]

    def step(self, inputs, h_pre, w_pre, k_pre, seq_cond, seq_cond_mask,
             mask=None):

        inputs = T.concatenate([inputs, w_pre], axis=1)

        h = self.layer.step(inputs, h_pre, mask=mask, process_inputs=True)

        act = T.exp(T.dot(h, self.w_cond) + self.b_cond)

        a = act[:, :self.n_mixt]
        b = act[:, self.n_mixt:2*self.n_mixt]
        k = k_pre + 0.1*act[:, -self.n_mixt:]

        u = T.shape_padright(T.arange(seq_cond.shape[0], dtype=floatX), 2)
        phi = T.sum(a * T.exp(-b * (k-u)**2), axis=-1)
        phi = phi * seq_cond_mask

        w = T.sum(T.shape_padright(phi) * seq_cond, axis=0)

        if mask:
            w = mask[:, None]*w + (1-mask[:, None])*w_pre
            k = mask[:, None]*k + (1-mask[:, None])*k_pre

        w = grad_clip(w, -100, 100)

        return h, w, phi, k

    def apply(self, seq_inputs, seq_mask, seq_cond, seq_cond_mask,
              h_ini, w_ini, k_ini):

        def scan_step(inputs, mask, h_pre, w_pre, k_pre,
                      seq_cond, seq_cond_mask):
            return self.step(inputs, h_pre, w_pre, k_pre,
                             seq_cond, seq_cond_mask, mask)

        (seq_h, seq_w, _, seq_k), scan_updates = theano.scan(
            fn=scan_step,
            sequences=[seq_inputs, seq_mask],
            outputs_info=[h_ini, w_ini, None, k_ini],
            non_sequences=[seq_cond, seq_cond_mask]
        )

        return (seq_h, seq_w, seq_k), scan_updates
