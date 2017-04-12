import theano
from theano import shared
import theano.tensor as T
from theano.gradient import grad_clip

import numpy as np

from utils import to_list

theano.config.floatX = 'float32'
floatX = theano.config.floatX


def normal_mat(size):
    return np.random.normal(0, 0.001, size=size).astype(floatX)


class PositionAttentionMechanism:
    def __init__(self, n_in_cond, n_mixt, initializer, n_out,
                 position_gap=0.1, grad_clip=None, normalize_att=False,
                 softmax_phi=False):
        self.n_in_cond = n_in_cond
        self.n_mixt = n_mixt
        self.position_gap = position_gap
        self.grad_clip = grad_clip
        self.n_out = n_out
        self.normalize_att = normalize_att
        self.softmax_phi = softmax_phi

        self.w_cond = shared(initializer.sample((n_out, 3*n_mixt)), 'w_cond')
        self.b_cond = shared(normal_mat((3*n_mixt, )), 'b_cond')

        self.params = [self.w_cond, self.b_cond]

    def step(self, h, k_pre, w_pre, seq_cond, seq_cond_mask, mask=None):
        # act: (batch_size, 3*n_mixt)
        act = T.dot(h, self.w_cond) + self.b_cond

        if not self.normalize_att:
            act = T.exp(act)
            a = act[:, :self.n_mixt]
            b = act[:, self.n_mixt:2*self.n_mixt]
            k = k_pre + self.position_gap * act[:, -self.n_mixt:]

        else:
            a = T.nnet.softmax(act[:, :self.n_mixt])
            b = 2. + 2. * T.tanh(act[:, self.n_mixt:2 * self.n_mixt])
            k = k_pre + self.position_gap * (
                2. + 2. * T.tanh(act[:, self.n_mixt:2 * self.n_mixt]))

        k = k
        # u: (length_cond_sequence, 1, 1)
        u = T.shape_padright(T.arange(seq_cond.shape[0], dtype=floatX), 2)
        # phi: (length_cond_sequence, batch_size, n_mixt)
        phi = T.sum(a * T.exp(-b * (k-u)**2), axis=-1)
        # phi: (length_cond_sequence, batch_size)
        phi *= seq_cond_mask

        if self.softmax_phi:  # TODO (not in Graves)
            phi = phi * seq_cond_mask + -1000*(1-seq_cond_mask)
            phi = T.nnet.softmax(phi.T).T * seq_cond_mask

        # w: (batch_size, condition_n_features)
        w = T.sum(T.shape_padright(phi) * seq_cond, axis=0)

        if mask:
            k = mask[:, None]*k + (1-mask[:, None])*k_pre
            w = mask[:, None]*w + (1-mask[:, None])*w_pre

        if self.grad_clip:
            w = grad_clip(w, -self.grad_clip, self.grad_clip)

        return a, k, phi, w


class SimplePositionAttentionMechanism:
    """
    Same as Graves except that there is no mixture and a softmax over
    the sequence.
    """
    def __init__(self, n_in_cond, initializer, n_out,
                 position_gap=0.1, grad_clip=None):
        self.n_in_cond = n_in_cond
        self.n_mixt = 1
        self.position_gap = position_gap
        self.grad_clip = grad_clip
        self.n_out = n_out

        self.w_cond = shared(initializer.sample((n_out, 3)), 'w_cond')
        self.b_cond = shared(normal_mat((3, )), 'b_cond')

        self.params = [self.w_cond, self.b_cond]

    def step(self, h, k_pre, w_pre, seq_cond, seq_cond_mask, mask=None):
        # act: (batch_size, 3*n_mixt)
        act = T.exp(T.dot(h, self.w_cond) + self.b_cond)

        a = act[:, :self.n_mixt]
        b = act[:, self.n_mixt:2*self.n_mixt]
        k = k_pre + self.position_gap * act[:, -self.n_mixt:]

        # u: (length_cond_sequence, 1, 1)
        u = T.shape_padright(T.arange(seq_cond.shape[0], dtype=floatX), 1)
        # phi: (length_cond_sequence, batch_size, n_mixt)
        temp = ((-b[:, 0] * (k[:, 0] - u) ** 2) * seq_cond_mask -1000 * (
            1 - seq_cond_mask))
        phi = T.nnet.softmax(temp.T).T
        # phi: (length_cond_sequence, batch_size)
        phi *= seq_cond_mask

        # w: (batch_size, condition_n_features)
        w = T.sum(T.shape_padright(phi) * seq_cond, axis=0)

        if mask:
            k = mask[:, None]*k + (1-mask[:, None])*k_pre
            w = mask[:, None]*w + (1-mask[:, None])*w_pre

        if self.grad_clip:
            w = grad_clip(w, -self.grad_clip, self.grad_clip)

        return a, k, phi, w


class PositionAttentionLayer:
    """
    Positional attention mechanism as described by Alex Graves in
    http://arxiv.org/abs/1308.0850

    This layer can have several attention mechanisms.
    """
    def __init__(self, layer_to_be_conditioned, ls_attention_mechanisms):

        self.layer = layer_to_be_conditioned
        if not isinstance(ls_attention_mechanisms, (list, tuple)):
            ls_attention_mechanisms = [ls_attention_mechanisms]
        self.ls_mechanisms = ls_attention_mechanisms
        self.n_mechanisms = len(ls_attention_mechanisms)

        self.params = layer_to_be_conditioned.params
        for mech in ls_attention_mechanisms:
            self.params.extend(mech.params)

    def step(self, inputs, h_pre, mask=None, *args):
        """
        A single timestep.

        Parameters
        ----------
        inputs: (batch_size, n_in)
        h_pre: (batch_size, n_hidden)
        mask: (batch_size,)

        *args contain k*(k_pre, w_pre) and k*(seq_cond, seq_cond_mask) where k
            is the number of attention mechanisms.
        k_pre: (batch_size, n_mixt)
        w_pre: (batch_size, n_in_cond)
        seq_cond: (length_cond_sequence, batch_size, n_in_cond)
        seq_cond_mask: (length_cond_sequence, batch_size)
        """

        # inputs: (batch_size, n_in + n_in_cond)
        all_w_pre = [args[2*i+1] for i in range(self.n_mechanisms)]
        inputs = T.concatenate([inputs] + all_w_pre, axis=1)

        # h: (batch_size, n_hidden)
        h = self.layer.step(inputs, h_pre, mask=mask, process_inputs=True)

        out_att = [h]
        offset = 2*self.n_mechanisms
        for i, mech in enumerate(self.ls_mechanisms):
            k_pre, w_pre = args[2*i: 2*i + 2]
            seq_cond, seq_cond_mask = args[offset + 2*i: offset + 2*i + 2]
            a, k, phi, w = self.ls_mechanisms[i].step(
                h, k_pre, w_pre, seq_cond, seq_cond_mask, mask=mask)
            out_att.extend([a, k, phi, w])

        # h, a, k, phi, w
        return tuple(out_att)

    def apply(self, seq_inputs, seq_mask, ls_seq_cond, ls_seq_cond_mask,
              h_ini, ls_k_ini, ls_w_ini):

        ls_seq_cond = to_list(ls_seq_cond)
        ls_seq_cond_mask = to_list(ls_seq_cond_mask)
        ls_k_ini = to_list(ls_k_ini)
        ls_w_ini = to_list(ls_w_ini)

        def scan_step(inputs, mask, h_pre, *args):
            # h, a, k, phi, w
            return self.step(inputs, h_pre, mask, *args)

        outputs_info, non_sequences = [h_ini], []
        for i in range(self.n_mechanisms):
            outputs_info.extend([None, ls_k_ini[i], None, ls_w_ini[i]])
            non_sequences.extend([ls_seq_cond[i], ls_seq_cond_mask[i]])

        scan_outputs, scan_updates = theano.scan(
            fn=scan_step,
            sequences=[seq_inputs, seq_mask],
            outputs_info=outputs_info,
            non_sequences=non_sequences
        )

        # scan_outputs is like (h, a, k, phi, w) with possibly several
        # attentions
        seq_h = scan_outputs[0]
        ls_seq_a, ls_seq_k, ls_seq_p, ls_seq_w = [], [], [], []
        for i in range(self.n_mechanisms):
            ls_seq_a.append(scan_outputs[4*i + 1])
            ls_seq_k.append(scan_outputs[4*i + 2])
            ls_seq_p.append(scan_outputs[4*i + 3])
            ls_seq_w.append(scan_outputs[4*i + 4])

        return (seq_h, ls_seq_a, ls_seq_k, ls_seq_p, ls_seq_w), scan_updates


class AttentionLayerNaive():
    def __init__(self, activation=T.nnet.softmax):
        self.activation = activation

    def apply(self, seq, seq_mask, attention_weights):
        """
        Parameters
        ----------
        seq: (seq_length, batch_size, n_hidden)
        seq_mask: (seq_length, batch_size)
        attention_weights: (length_seq_text, batch_size, 1)

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
