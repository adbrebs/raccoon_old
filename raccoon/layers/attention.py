import theano
from theano import shared
import theano.tensor as T
from theano.gradient import grad_clip

import numpy as np

theano.config.floatX = 'float32'
floatX = theano.config.floatX


def normal_mat(size):
    return np.random.normal(0, 0.001, size=size).astype(floatX)


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

    def step(self, inputs, h_pre, k_pre, w_pre, seq_cond, seq_cond_mask,
             mask=None):
        """
        A single timestep.

        Parameters
        ----------
        inputs: (batch_size, n_in)
        h_pre: (batch_size, n_hidden)
        mask: (batch_size,)

        k_pre: (batch_size, n_mixt)
        w_pre: (batch_size, n_in_cond)

        seq_cond: (length_cond_sequence, batch_size, n_in_cond)
        seq_cond_mask: (length_cond_sequence, batch_size)
        """
        # inputs: (batch_size, n_in + n_in_cond)
        inputs = T.concatenate([inputs, w_pre], axis=1)

        # h: (batch_size, n_hidden)
        h = self.layer.step(inputs, h_pre, mask=mask, process_inputs=True)

        # act: (batch_size, 3*n_mixt)
        act = T.exp(T.dot(h, self.w_cond) + self.b_cond)

        a = act[:, :self.n_mixt]
        b = act[:, self.n_mixt:2*self.n_mixt]
        k = k_pre + 0.1*act[:, -self.n_mixt:]

        # u: (length_cond_sequence, 1, 1)
        u = T.shape_padright(T.arange(seq_cond.shape[0], dtype=floatX), 2)
        # phi: (length_cond_sequence, batch_size, n_mixt)
        phi = T.sum(a * T.exp(-b * (k-u)**2), axis=-1)
        # phi: (length_cond_sequence, batch_size)
        phi = phi * seq_cond_mask

        # w: (batch_size, n_chars)
        w = T.sum(T.shape_padright(phi) * seq_cond, axis=0)

        if mask:
            k = mask[:, None]*k + (1-mask[:, None])*k_pre
            w = mask[:, None]*w + (1-mask[:, None])*w_pre

        w = grad_clip(w, -100, 100)

        return h, a, k, phi, w

    def apply(self, seq_inputs, seq_mask, seq_cond, seq_cond_mask,
              h_ini, k_ini, w_ini):

        def scan_step(inputs, mask, h_pre, k_pre, w_pre,
                      seq_cond, seq_cond_mask):
            return self.step(inputs, h_pre, k_pre, w_pre,
                             seq_cond, seq_cond_mask, mask)

        (seq_h, _, seq_k, _, seq_w), scan_updates = theano.scan(
            fn=scan_step,
            sequences=[seq_inputs, seq_mask],
            outputs_info=[h_ini, None, k_ini, None, w_ini],
            non_sequences=[seq_cond, seq_cond_mask]
        )

        return (seq_h, seq_k, seq_w), scan_updates


class AttentionLayerNaive():
    def __init__(self, activation=T.nnet.softmax):
        self.activation = activation

    def apply(self, seq, seq_mask, attention_weights):
        """
        Parameters
        ----------
        seq: (seq_length, batch_size, n_hidden)
        seq_mask: (seq_length, batch_size)
        attention_weights: (length_seq_context, batch_size, 1)

        Returns
        -------
        (batch_size, n_hidden)
        """
        att = T.shape_padright(attention_weights)
        att = T.exp(att - T.max(att, axis=0, keepdims=True))
        att = att * seq_mask[:, :, None]
        att /= T.sum(att, axis=0, keepdims=True)

        return (att * seq).sum(axis=0)


class AttentionLayerEfficient():
    def __init__(self):
        pass

    def apply(self, covariance, condition):
        """
        Parameters
        ----------
        covariance: (batch_size, n_hidden, n_hidden)
        condition: (batch_size, n_hidden)

        Returns
        -------
        (batch_size, n_hidden)
        """
        return T.sum(covariance * condition.dimshuffle((0, 'x', 1)), axis=2)
