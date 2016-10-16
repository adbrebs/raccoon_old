import theano
from theano import shared
import theano.tensor as T

import numpy as np

from lasagne import init
from lasagne.layers.recurrent import Gate
from lasagne import nonlinearities

theano.config.floatX = 'float32'
floatX = theano.config.floatX


class GRUUnit:
    def __init__(self, n_in, n_out,
                 resetgate=Gate(W_cell=None),
                 updategate=Gate(W_cell=None),
                 hidden_update=Gate(W_cell=None, nonlinearity=nonlinearities.tanh),
                 grad_clipping=100):
        self.n_in = n_in
        self.n_out = n_out
        self.grad_clipping = grad_clipping

        self.params = []

        def create_gate_params(gate):
            """ Convenience function for adding layer parameters from a Gate
            instance. """
            return (gate.W_in.sample((n_in, n_out)),
                    gate.W_hid.sample((n_out, n_out)),
                    gate.b.sample((n_out,)),
                    gate.nonlinearity)

        (W_in_to_updategate, W_hid_to_updategate, b_updategate,
         self.nonlinearity_updategate) = create_gate_params(updategate)

        (W_in_to_resetgate, W_hid_to_resetgate, b_resetgate,
         self.nonlinearity_resetgate) = create_gate_params(resetgate)

        (W_in_to_hidden_update, W_hid_to_hidden_update, b_hidden_update,
         self.nonlinearity_hid) = create_gate_params(hidden_update)

        W_in_stacked = np.concatenate(
            [W_in_to_resetgate, W_in_to_updategate, W_in_to_hidden_update], axis=1)
        self.W_in_stacked = shared(W_in_stacked, 'W_in')

        W_hid_stacked = np.concatenate(
            [W_hid_to_resetgate, W_hid_to_updategate, W_hid_to_hidden_update], axis=1)
        self.W_hid_stacked = shared(W_hid_stacked, 'W_hid')

        b_stacked = np.concatenate(
            [b_resetgate, b_updategate, b_hidden_update], axis=0)
        self.b_stacked = shared(b_stacked, name='b')

        self.in_params = [self.W_in_stacked, self.b_stacked]
        self.rec_params = [self.W_hid_stacked]

    def precompute_inputs(self, inputs):
        return T.dot(inputs, self.W_in_stacked) + self.b_stacked

    def step(self, input_n, mask_n, hid_previous, process_inputs=False):

        def slice_w(x, n):
            return x[:, n * self.n_out:(n + 1) * self.n_out]

        hid_input = T.dot(hid_previous, self.W_hid_stacked)

        if self.grad_clipping:
            input_n = theano.gradient.grad_clip(
                input_n, -self.grad_clipping, self.grad_clipping)
            hid_input = theano.gradient.grad_clip(
                hid_input, -self.grad_clipping, self.grad_clipping)

        if process_inputs:
            # Compute W_{xr}x_t + b_r, W_{xu}x_t + b_u, and W_{xc}x_t + b_c
            input_n = self.precompute_inputs(input_n)

        # Reset and update gates
        resetgate = slice_w(hid_input, 0) + slice_w(input_n, 0)
        updategate = slice_w(hid_input, 1) + slice_w(input_n, 1)
        resetgate = self.nonlinearity_resetgate(resetgate)
        updategate = self.nonlinearity_updategate(updategate)

        # Compute W_{xc}x_t + r_t \odot (W_{hc} h_{t - 1})
        hidden_update_in = slice_w(input_n, 2)
        hidden_update_hid = slice_w(hid_input, 2)
        hidden_update = hidden_update_in + resetgate * hidden_update_hid
        if self.grad_clipping:
            hidden_update = theano.gradient.grad_clip(
                hidden_update, -self.grad_clipping, self.grad_clipping)
        hidden_update = self.nonlinearity_hid(hidden_update)

        # Compute (1 - u_t)h_{t - 1} + u_t c_t
        hid = (1 - updategate) * hid_previous + updategate * hidden_update

        hid = T.switch(mask_n[:, None], hid, hid_previous)
        return hid


class CovarianceMatrixUpdateRule:
    """
    Class that defines how the covariance matrix should be updated.
    """
    def __init__(self, decay_l=1.0, decay_r=1.0, normalize_length_seq=True):
        self.params = []
        self.decay_l = decay_l
        self.decay_r = decay_r
        self.normalize_length_seq = normalize_length_seq

    def update_matrix(self, timestep, h, mask, C_pre):
        h = T.switch(mask[:, None], h, .0)

        update = self.compute_update(timestep, h, C_pre)

        update_square = update.dimshuffle(0, 1, 'x') * update.dimshuffle(0, 'x', 1)

        return self.decay_l * C_pre + self.decay_r * update_square, update

    def compute_update(self, timestep, h, C_pre):
        raise NotImplemented

    def restore_previous_matrix(self, C, update):
        update_square = update.dimshuffle(0, 1, 'x') * update.dimshuffle(0, 'x', 1)
        C_pre = (C - self.decay_r * update_square) / self.decay_l
        return C_pre

    def build_backward_graph(self):
        h = T.matrix()
        mask = T.vector()
        C_pre = T.tensor3()
        grad_att = T.matrix()
        query = T.matrix()

        C, _ = self.update_matrix(h, mask, C_pre)
        att = T.sum(C * query.dimshuffle(0, 'x', 1), axis=-1)

        grads = T.grad(None, [h] + self.params,
                       known_grads={att: grad_att})

        grad_h = grads[0]
        grad_params = grads[1:]

        return h, mask, C_pre, grad_att, query, (grad_h, grad_params)


class BasicCovariance(CovarianceMatrixUpdateRule):
    """
    Computes the real covariance matrix.
    """
    def __init__(self, decay_l=1.0, decay_r=1.0, normalize_length_seq=True):
        CovarianceMatrixUpdateRule.__init__(
            self, decay_l=decay_l, decay_r=decay_r,
            normalize_length_seq=normalize_length_seq)

    def compute_update(self, timestep, h, C_pre):
        return h


class GatedCovariance(CovarianceMatrixUpdateRule):
    """
    The update is the state times a sigmoid vector. The network can decide
    which features to update in the covariance matrix. It bases its decision
    on the current state h and on T.dot(h, C).
    """
    def __init__(self, n_hidden=100):
        CovarianceMatrixUpdateRule.__init__(self)
        ini = init.Constant(0.0)
        self.n_hidden = n_hidden
        self.W1 = shared(ini.sample((n_hidden, n_hidden)))
        self.W2 = shared(ini.sample((n_hidden, n_hidden)))
        self.b = shared(init.Constant(1.0)((n_hidden,)))
        self.params = [self.W1, self.W2, self.b]

    def compute_update(self, timestep, h, C_pre):
        a = 1000*T.sum(C_pre * h.dimshuffle(0, 'x', 1), axis=-1) / (timestep + 1)
        b = h
        alpha = T.nnet.sigmoid(T.dot(a, self.W1) + T.dot(b, self.W2) + self.b)
        update = h * alpha
        return update


class EfficientAttentionDesign:
    """
    Recurrent layer that exploits the covariance matrix of the hidden states.
    """
    def __init__(self, recurrent_fun, rec_params, attention_update_rule):
        """
        Parameters
        ----------
        recurrent_fun: function
            Takes as input (input, mask, h_pre) and returns the current state h
        rec_params: list of shared variables
            Trainable params used in the recurrent_fun
        attention_update_rule: CovarianceMatrixUpdateRule object
            Object which controls how to update the covariance matrix
        """
        self.recurrent_fun = recurrent_fun
        self.rec_params = rec_params
        self.attention_update_rule = attention_update_rule

    def forward(self, seq_inputs, query, h_ini, C_ini, seq_mask=None):
        """
        Recurse over the whole sequences

        Parameters
        ----------
        seq_inputs: (length_sequence, batch_size, n_in)
        seq_mask: (length_sequence, batch_size)
        query: (bs, n_hidden)
        h_ini: (bs, n_hidden)
        C_ini: (bs, n_hidden, n_hidden)

        Returns
        -------
        Tuple of
            seq_h: (length_sequence, batch_size, n_hidden)
            seq_updates: (length_sequence, batch_size, n_hidden)
            att: (batch_size, n_hidden)
            cov: (batch_size, n_hidden, n_hidden)
        """
        if not seq_mask:
            seq_mask = T.ones(seq_inputs.shape[:-1])

        def step(timestep, inputs, mask, h_pre, C_pre):
            """
            A single timestep.

            Parameters
            ----------
            inputs: (batch_size, n_in)
            mask: (batch_size,)
            h_pre: (batch_size, n_hidden)
            C_pre: (batch_size, n_hidden, n_hidden)
            """
            h = self.recurrent_fun(inputs, mask, h_pre)

            C, update = self.attention_update_rule.update_matrix(
                timestep, h, mask, C_pre)

            return h, update, C

        timesteps = T.arange(0, seq_inputs.shape[0], dtype=floatX)

        (seq_h, seq_updates, seq_C), _ = theano.scan(
            fn=step,
            sequences=[timesteps, seq_inputs, seq_mask],
            outputs_info=[h_ini, None, C_ini],
            name='step')

        cov = seq_C[-1]

        att = 1000 * T.sum(cov * query.dimshuffle(0, 'x', 1), axis=-1)
        att /= T.sum(seq_mask, axis=0)[:, None]

        return seq_h, seq_updates, att, cov

    def build_backward_rec_graph(self):
        """
        Builds the backward graph of the recurrence operation defined by
        self.recurrent_fun.
        """
        h_pre = T.matrix()
        grad_h = T.matrix()
        input = T.matrix()
        mask = T.vector()

        back_h = self.recurrent_fun(input, mask, h_pre)

        grads = T.grad(
            None, [input, h_pre] + self.rec_params,
            known_grads={back_h: grad_h})

        grad_input = grads[0]
        grad_h_pre = grads[1]
        grad_params = grads[2:]

        return input, mask, h_pre, grad_h, grad_input, grad_h_pre, grad_params

    def backward(self, seq_inputs, seq_h, seq_updates,
                 query, grad_att, cov, seq_mask=None):
        """
        Parameters
        ----------
        seq_inputs: (length_seq, bs, n_features)
        seq_mask: (length_seq, bs)
        seq_h: (length_seq, bs, n_hidden)
        seq_updates: (length_seq, bs, n_hidden)
            the update vectors that have been added to the covariance matrix at
            each timestep in the step pass.
        query: (bs, n_hidden)
        grad_att: (bs, n_hidden)
            Gradient of the loss with respect to T.dot(cov, query)
        cov: (bs, n_hidden, n_hidden)
            Covariance matrix computed in the step pass
        """
        if not seq_mask:
            seq_mask = T.ones(seq_inputs.shape[:-1])

        # Build backward graph of the recurrence
        (back_input, back_mask, back_h_pre, back_grad_h, back_grad_input,
         back_grad_h_pre, back_grad_params) = self.build_backward_rec_graph()

        # Build backward graph of the attention update rule
        (u_h, u_mask, u_C_pre, u_grad_att, u_query,
         (u_grad_h, u_grad_params)) = self.attention_update_rule.build_backward_graph()

        def step(input, mask, h, h_pre, update, grad_h, C, *args):
            """
            A single timestep of the backward pass.

            Parameters
            ----------
            input: (batch_size, n_in)
            mask: (batch_size,)
            h: (batch_size, n_hidden)
            h_pre: (batch_size, n_hidden)
            update: (batch_size, n_hidden)
            grad_h: (batch_size, n_hidden)
            C: (batch_size, n_hidden, n_hidden)

            args
                args[-2] query: (batch_size, n_hidden)
                args[-1] grad_att: (batch_size, n_hidden)
                args[:-2] grad_params

            Returns
            -------
            grad_input: (batch_size, n_in)
            grad_h_pre: (batch_size, n_hidden)
            C_pre: (batch_size, n_hidden, n_hidden)
            gradients with respect to the params (both of the recurrent and the
             update rule)
            """
            query, grad_att = args[-2:]
            prev_grad_params = args[:-2]

            C_pre = self.attention_update_rule.restore_previous_matrix(C, update)

            att_grads = theano.clone(
                output=[u_grad_h] + u_grad_params,
                replace={u_h: h,
                         u_mask: mask,
                         u_C_pre: C_pre,
                         u_grad_att: grad_att,
                         u_query: query})

            grad_h_att = att_grads[0]
            grad_params_att = att_grads[1:]

            grad_h_att *= 1000 / T.sum(seq_mask, axis=0)[:, None]
            grad_h_att = T.switch(mask[:, None], grad_h_att, .0)

            rec_grads = theano.clone(
                output=[back_grad_input, back_grad_h_pre] + back_grad_params,
                replace={back_input: input,
                         back_mask: mask,
                         back_h_pre: h_pre,
                         back_grad_h: grad_h + grad_h_att})

            grad_input = rec_grads[0]
            grad_h_pre = rec_grads[1]
            grad_params_rec = rec_grads[2:]

            grad_params = grad_params_att + grad_params_rec
            scan_outputs = [grad_input, grad_h_pre, C_pre]
            for prev_grad, grad in zip(prev_grad_params, grad_params):
                scan_outputs.append(prev_grad + grad)

            return tuple(scan_outputs)

        seq_h = T.concatenate([T.zeros_like(seq_h[0:1]), seq_h])

        params = self.attention_update_rule.params + self.rec_params
        grads, _ = theano.scan(
            fn=step,
            sequences=[seq_inputs[::-1], seq_mask[::-1],
                       dict(input=seq_h[::-1], taps=[0, 1]),
                       seq_updates[::-1]],
            outputs_info=([None, T.zeros_like(grad_att), cov] +
                          [T.zeros_like(m) for m in params]),
            non_sequences=[query, grad_att], name='backward')
        grads_input = grads[0][::-1]
        grads_param = [g[-1] for g in grads[3:]]

        return grads_input, params, grads_param


class MultiEfficientAttentionDesign:
    """
    Recurrent layer that exploits the covariance matrix of the hidden states.
    """
    def __init__(self, recurrent_fun, rec_params, attention_update_rule):
        """
        Parameters
        ----------
        recurrent_fun: function
            Takes as input (input, mask, h_pre) and returns the current state h
        rec_params: list of shared variables
            Trainable params used in the recurrent_fun
        attention_update_rule: CovarianceMatrixUpdateRule object
            Object which controls how to update the covariance matrix
        return_every_timestep: bool
            If True, will return an attention vector at each timestep. If
            false, will only return a single attention vector corresponding to
            the last state
        """
        self.recurrent_fun = recurrent_fun
        self.rec_params = rec_params
        self.attention_update_rule = attention_update_rule

    def forward(self, seq_inputs, h_ini, C_ini, query=None, seq_mask=None):
        """
        Recurse over the whole sequences

        Parameters
        ----------
        seq_inputs: (length_sequence, batch_size, n_in)
        seq_mask: (length_sequence, batch_size)
        query: (bs, n_hidden)
        h_ini: (bs, n_hidden)
        C_ini: (bs, n_hidden, n_hidden)

        Returns
        -------
        Tuple of
            seq_h: (length_sequence, batch_size, n_hidden)
            seq_updates: (length_sequence, batch_size, n_hidden)
            att: (batch_size, n_hidden)
            cov: (batch_size, n_hidden, n_hidden)
        """
        if not seq_mask:
            seq_mask = T.ones(seq_inputs.shape[:-1])

        def step(inputs, mask, h_pre, C_pre):
            """
            A single timestep.

            Parameters
            ----------
            inputs: (batch_size, n_in)
            mask: (batch_size,)
            h_pre: (batch_size, n_hidden)
            C_pre: (batch_size, n_hidden, n_hidden)
            """
            h = self.recurrent_fun(inputs, mask, h_pre)

            C, update = self.attention_update_rule.update_matrix(
                h, mask, C_pre)

            if query:
                att = T.sum(C * query.dimshuffle(0, 'x', 1), axis=-1)
            else:
                att = T.sum(C * h.dimshuffle(0, 'x', 1), axis=-1)

            return [h, update, C, att]

        ret_scan, _ = theano.scan(
            fn=step,
            sequences=[seq_inputs, seq_mask],
            outputs_info=[h_ini, None, C_ini, None],
            name='step')

        seq_h, seq_updates, seq_C, seq_att = ret_scan
        cov = seq_C[-1]
        return seq_h, seq_updates, seq_att, cov

    def build_backward_rec_graph(self):
        """
        Builds the backward graph of the recurrence operation defined by
        self.recurrent_fun.
        """
        h_pre = T.matrix()
        grad_h = T.matrix()
        input = T.matrix()
        mask = T.vector()

        back_h = self.recurrent_fun(input, mask, h_pre)

        grads = T.grad(
            None, [input, h_pre] + self.rec_params,
            known_grads={back_h: grad_h})

        grad_input = grads[0]
        grad_h_pre = grads[1]
        grad_params = grads[2:]

        return input, mask, h_pre, grad_h, grad_input, grad_h_pre, grad_params

    def backward(self, seq_inputs, seq_h, seq_updates,
                 grad_seq_att, cov, query=None, extra_grad_seq_h=None,
                 seq_mask=None):
        """
        Parameters
        ----------
        seq_inputs: (length_seq, bs, n_features)
        seq_mask: (length_seq, bs)
        seq_h: (length_seq, bs, n_hidden)
        seq_updates: (length_seq, bs, n_hidden)
            the update vectors that have been added to the covariance matrix at
            each timestep in the step pass.
        query: (bs, n_hidden) or None
        grad_seq_att: (length_seq, bs, n_hidden)
            Gradient of the loss with respect to T.dot(cov, query)
        cov: (bs, n_hidden, n_hidden)
            Covariance matrix computed in the step pass
        """
        if not seq_mask:
            seq_mask = T.ones(seq_inputs.shape[:-1])

        if not extra_grad_seq_h:
            extra_grad_seq_h = T.zeros(grad_seq_att.shape)

        # Build backward graph of the recurrence
        (back_input, back_mask, back_h_pre, back_grad_h, back_grad_input,
         back_grad_h_pre, back_grad_params) = self.build_backward_rec_graph()

        # Build backward graph of the attention update rule
        (u_h, u_mask, u_C_pre, u_grad_att, u_query,
         (u_grad_h, u_grad_params)) = self.attention_update_rule.build_backward_graph()

        cumsum_grad_seq_att = T.cumsum(grad_seq_att[::-1], axis=0)

        def step(input, mask, cumsum_grad_att, extra_grad_h, h, h_pre, update, grad_h, C,
                 *prev_grad_params):
            """
            A single timestep of the backward pass.

            Parameters
            ----------
            input: (batch_size, n_in)
            mask: (batch_size,)
            cumsum_grad_att: (batch_size, n_hidden)
            h: (batch_size, n_hidden)
            h_pre: (batch_size, n_hidden)
            update: (batch_size, n_hidden)
            grad_h: (batch_size, n_hidden)
            C: (batch_size, n_hidden, n_hidden)
            *prev_grad_params

            Returns
            -------
            grad_input: (batch_size, n_in)
            grad_h_pre: (batch_size, n_hidden)
            C_pre: (batch_size, n_hidden, n_hidden)
            gradients with respect to the params (both of the recurrent and the
             update rule)
            """
            C_pre = self.attention_update_rule.restore_previous_matrix(C, update)

            att_grads = theano.clone(
                output=[u_grad_h] + u_grad_params,
                replace={u_h: h,
                         u_mask: mask,
                         u_C_pre: C_pre,
                         u_grad_att: cumsum_grad_att,
                         u_query: h})

            grad_h_att = att_grads[0]
            grad_params_att = att_grads[1:]

            grad_h_att *= 1000 / T.sum(seq_mask, axis=0)[:, None]
            grad_h_att = T.switch(mask[:, None], grad_h_att, .0)

            rec_grads = theano.clone(
                output=[back_grad_input, back_grad_h_pre] + back_grad_params,
                replace={back_input: input,
                         back_mask: mask,
                         back_h_pre: h_pre,
                         back_grad_h: extra_grad_h + grad_h + grad_h_att})

            grad_input = rec_grads[0]
            grad_h_pre = rec_grads[1]
            grad_params_rec = rec_grads[2:]

            grad_params = grad_params_att + grad_params_rec
            scan_outputs = [grad_input, grad_h_pre, C_pre]
            for prev_grad, grad in zip(prev_grad_params, grad_params):
                scan_outputs.append(prev_grad + grad)

            return tuple(scan_outputs)

        seq_h = T.concatenate([T.zeros_like(seq_h[0:1]), seq_h])

        params = self.attention_update_rule.params + self.rec_params
        grads, _ = theano.scan(
            fn=step,
            sequences=[seq_inputs[::-1], seq_mask[::-1], cumsum_grad_seq_att,
                       extra_grad_seq_h[::-1],
                       dict(input=seq_h[::-1], taps=[0, 1]),
                       seq_updates[::-1]],
            outputs_info=([None, T.zeros_like(seq_h[0]), cov] +
                          [T.zeros_like(m) for m in params]),
            name='backward')
        grads_input = grads[0][::-1]
        grads_param = [g[-1] for g in grads[3:]]

        return grads_input, params, grads_param
