from collections import OrderedDict

import numpy as np

import theano
import theano.tensor as T

from lasagne.layers import (
    MergeLayer, Layer, ElemwiseMergeLayer, dimshuffle, ReshapeLayer,
    DenseLayer, ElemwiseSumLayer, concat, DropoutLayer)
from lasagne.layers.recurrent import Gate
from lasagne import nonlinearities
from lasagne import init
from lasagne.utils import unroll_scan

floatX = theano.config.floatX = 'float32'



class GRULayer(MergeLayer):
    def __init__(self, incoming, num_units,
                 resetgate=Gate(W_cell=None),
                 updategate=Gate(W_cell=None),
                 hidden_update=Gate(W_cell=None,
                                    nonlinearity=nonlinearities.tanh),
                 hid_init=init.Constant(0.),
                 cov_init=init.Constant(0.),
                 backwards=False,
                 learn_init=False,
                 gradient_steps=-1,
                 grad_clipping=0,
                 unroll_scan=False,
                 precompute_input=True,
                 mask_input=None,
                 only_return_final=False,
                 **kwargs):

        # This layer inherits from a MergeLayer, because it can have three
        # inputs - the layer input, the mask and the initial hidden state.  We
        # will just provide the layer input as incomings, unless a mask input
        # or initial hidden state was provided.
        incomings = [incoming]
        self.mask_incoming_index = -1
        self.hid_init_incoming_index = -1
        self.cov_init_incoming_index = -1
        if mask_input is not None:
            incomings.append(mask_input)
            self.mask_incoming_index = len(incomings)-1
        if isinstance(hid_init, Layer):
            incomings.append(hid_init)
            self.hid_init_incoming_index = len(incomings)-1
        if isinstance(cov_init, Layer):
            incomings.append(cov_init)
            self.cov_init_incoming_index = len(incomings)-1

        # Initialize parent layer
        super(GRULayer, self).__init__(incomings, **kwargs)

        self.learn_init = learn_init
        self.num_units = num_units
        self.grad_clipping = grad_clipping
        self.backwards = backwards
        self.gradient_steps = gradient_steps
        self.unroll_scan = unroll_scan
        self.precompute_input = precompute_input
        self.only_return_final = only_return_final

        if unroll_scan and gradient_steps != -1:
            raise ValueError(
                "Gradient steps must be -1 when unroll_scan is true.")

        # Retrieve the dimensionality of the incoming layer
        input_shape = self.input_shapes[0]

        if unroll_scan and input_shape[1] is None:
            raise ValueError("Input sequence length cannot be specified as "
                             "None when unroll_scan is True")

        # Input dimensionality is the output dimensionality of the input layer
        num_inputs = np.prod(input_shape[2:])

        def add_gate_params(gate, gate_name):
            """ Convenience function for adding layer parameters from a Gate
            instance. """
            return (self.add_param(gate.W_in, (num_inputs, num_units),
                                   name="W_in_to_{}".format(gate_name)),
                    self.add_param(gate.W_hid, (num_units, num_units),
                                   name="W_hid_to_{}".format(gate_name)),
                    self.add_param(gate.b, (num_units,),
                                   name="b_{}".format(gate_name),
                                   regularizable=False),
                    gate.nonlinearity)

        # Add in all parameters from gates
        (self.W_in_to_updategate, self.W_hid_to_updategate, self.b_updategate,
         self.nonlinearity_updategate) = add_gate_params(updategate,
                                                         'updategate')
        (self.W_in_to_resetgate, self.W_hid_to_resetgate, self.b_resetgate,
         self.nonlinearity_resetgate) = add_gate_params(resetgate, 'resetgate')

        (self.W_in_to_hidden_update, self.W_hid_to_hidden_update,
         self.b_hidden_update, self.nonlinearity_hid) = add_gate_params(
             hidden_update, 'hidden_update')

        # Initialize hidden state
        if isinstance(hid_init, Layer):
            self.hid_init = hid_init
        else:
            self.hid_init = self.add_param(
                hid_init, (1, self.num_units), name="hid_init",
                trainable=learn_init, regularizable=False)

        if isinstance(cov_init, Layer):
            self.cov_init = cov_init
        else:
            self.cov_init = self.add_param(
                cov_init, (1, self.num_units), name="cov_init",
                trainable=learn_init, regularizable=False)

    def get_output_shape_for(self, input_shapes):
        # The shape of the input to this layer will be the first element
        # of input_shapes, whether or not a mask input is being used.
        input_shape = input_shapes[0]
        # When only_return_final is true, the second (sequence step) dimension
        # will be flattened
        if self.only_return_final:
            return input_shape[0], self.num_units
        # Otherwise, the shape will be (n_batch, n_steps, num_units)
        else:
            return input_shape[0], input_shape[1], self.num_units

    def get_output_for(self, inputs, **kwargs):
        """
        Compute this layer's output function given a symbolic input variable

        Parameters
        ----------
        inputs : list of theano.TensorType
            `inputs[0]` should always be the symbolic input variable.  When
            this layer has a mask input (i.e. was instantiated with
            `mask_input != None`, indicating that the lengths of sequences in
            each batch vary), `inputs` should have length 2, where `inputs[1]`
            is the `mask`.  The `mask` should be supplied as a Theano variable
            denoting whether each time step in each sequence in the batch is
            part of the sequence or not.  `mask` should be a matrix of shape
            ``(n_batch, n_time_steps)`` where ``mask[i, j] = 1`` when ``j <=
            (length of sequence i)`` and ``mask[i, j] = 0`` when ``j > (length
            of sequence i)``. When the hidden state of this layer is to be
            pre-filled (i.e. was set to a :class:`Layer` instance) `inputs`
            should have length at least 2, and `inputs[-1]` is the hidden state
            to prefill with.

        Returns
        -------
        layer_output : theano.TensorType
            Symbolic output variable.
        """
        # Retrieve the layer input
        input = inputs[0]
        # Retrieve the mask when it is supplied
        mask = None
        hid_init = None
        cov_init = None
        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]
        if self.hid_init_incoming_index > 0:
            hid_init = inputs[self.hid_init_incoming_index]
        if self.cov_init_incoming_index > 0:
            cov_init = inputs[self.cov_init_incoming_index]

        # Treat all dimensions after the second as flattened feature dimensions
        if input.ndim > 3:
            input = T.flatten(input, 3)

        # Because scan iterates over the first dimension we dimshuffle to
        # (n_time_steps, n_batch, n_features)
        input = input.dimshuffle(1, 0, 2)
        seq_len, num_batch, _ = input.shape

        # Stack input weight matrices into a (num_inputs, 3*num_units)
        # matrix, which speeds up computation
        W_in_stacked = T.concatenate(
            [self.W_in_to_resetgate, self.W_in_to_updategate,
             self.W_in_to_hidden_update], axis=1)

        # Same for hidden weight matrices
        W_hid_stacked = T.concatenate(
            [self.W_hid_to_resetgate, self.W_hid_to_updategate,
             self.W_hid_to_hidden_update], axis=1)

        # Stack gate biases into a (3*num_units) vector
        b_stacked = T.concatenate(
            [self.b_resetgate, self.b_updategate,
             self.b_hidden_update], axis=0)

        if self.precompute_input:
            # precompute_input inputs*W. W_in is (n_features, 3*num_units).
            # input is then (n_batch, n_time_steps, 3*num_units).
            input = T.dot(input, W_in_stacked) + b_stacked

        # At each call to scan, input_n will be (n_time_steps, 3*num_units).
        # We define a slicing function that extract the input to each GRU gate
        def slice_w(x, n):
            return x[:, n*self.num_units:(n+1)*self.num_units]

        # Create single recurrent computation step function
        # input__n is the n'th vector of the input
        def step(input_n, hid_previous, cov_previous, *args):
            # Compute W_{hr} h_{t - 1}, W_{hu} h_{t - 1}, and W_{hc} h_{t - 1}
            hid_input = T.dot(hid_previous, W_hid_stacked)

            if self.grad_clipping:
                input_n = theano.gradient.grad_clip(
                    input_n, -self.grad_clipping, self.grad_clipping)
                hid_input = theano.gradient.grad_clip(
                    hid_input, -self.grad_clipping, self.grad_clipping)

            if not self.precompute_input:
                # Compute W_{xr}x_t + b_r, W_{xu}x_t + b_u, and W_{xc}x_t + b_c
                input_n = T.dot(input_n, W_in_stacked) + b_stacked

            # Reset and update gates
            resetgate = slice_w(hid_input, 0) + slice_w(input_n, 0)
            updategate = slice_w(hid_input, 1) + slice_w(input_n, 1)
            resetgate = self.nonlinearity_resetgate(resetgate)
            updategate = self.nonlinearity_updategate(updategate)

            # Compute W_{xc}x_t + r_t \odot (W_{hc} h_{t - 1})
            hidden_update_in = slice_w(input_n, 2)
            hidden_update_hid = slice_w(hid_input, 2)
            hidden_update = hidden_update_in + resetgate*hidden_update_hid
            if self.grad_clipping:
                hidden_update = theano.gradient.grad_clip(
                    hidden_update, -self.grad_clipping, self.grad_clipping)
            hidden_update = self.nonlinearity_hid(hidden_update)

            # Compute (1 - u_t)h_{t - 1} + u_t c_t
            hid = (1 - updategate)*hid_previous + updategate*hidden_update

            cov = cov_previous + hid.dimshuffle((0, 'x', 1)) * hid.dimshuffle((0, 1, 'x'))

            return hid, cov

        def step_masked(input_n, mask_n, hid_previous, cov_previous, *args):
            hid, cov = step(input_n, hid_previous, cov_previous, *args)

            # Skip over any input with mask 0 by copying the previous
            # hidden state; proceed normally for any input with mask 1.
            hid = T.switch(mask_n, hid, hid_previous)

            cov = T.switch(mask_n, cov, hid_previous)

            return hid, cov

        if mask is not None:
            # mask is given as (batch_size, seq_len). Because scan iterates
            # over first dimension, we dimshuffle to (seq_len, batch_size) and
            # add a broadcastable dimension
            mask = mask.dimshuffle(1, 0, 'x')
            sequences = [input, mask]
            step_fun = step_masked
        else:
            sequences = [input]
            step_fun = step

        if not isinstance(self.hid_init, Layer):
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            hid_init = T.dot(T.ones((num_batch, 1)), self.hid_init)

        # The hidden-to-hidden weight matrix is always used in step
        non_seqs = [W_hid_stacked]
        # When we aren't precomputing the input outside of scan, we need to
        # provide the input weights and biases to the step function
        if not self.precompute_input:
            non_seqs += [W_in_stacked, b_stacked]

        if self.unroll_scan:
            # Retrieve the dimensionality of the incoming layer
            input_shape = self.input_shapes[0]
            # Explicitly unroll the recurrence instead of using scan
            hid_out, cov_out = unroll_scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[hid_init, cov_init],
                go_backwards=self.backwards,
                non_sequences=non_seqs,
                n_steps=input_shape[1])[0]
        else:
            # Scan op iterates over first dimension of input and repeatedly
            # applies the step function
            hid_out, cov_out = theano.scan(
                fn=step_fun,
                sequences=sequences,
                go_backwards=self.backwards,
                outputs_info=[hid_init, cov_init],
                non_sequences=non_seqs,
                truncate_gradient=self.gradient_steps,
                strict=True)[0]

        # When it is requested that we only return the final sequence step,
        # we need to slice it out immediately after scan is applied
        if self.only_return_final:
            hid_out = hid_out[-1]
            cov_out = cov_out[-1]
        else:
            # dimshuffle back to (n_batch, n_time_steps, n_features))
            hid_out = hid_out.dimshuffle(1, 0, 2)
            cov_out = cov_out.dimshuffle(1, 0, 2, 3)

            # if scan is backward reverse the output
            if self.backwards:
                hid_out = hid_out[:, ::-1]
                cov_out = cov_out[:, ::-1]

        return hid_out, cov_out




class SequenceSoftmax(MergeLayer):
    """
    Computes a softmax over a sequence associated with a mask.

    Parameters
    ----------
    layer: layer of shape (batch_size, sequence_length, ...)
    layer_mask: layer of shape (batch_size, sequence_length, ...)

    Notes
    -----
    This layer has the same output shape as the parameter layer
    layer and layer_mask should have compatible shapes.
    """
    def __init__(self, layer, layer_mask, seq_axis=1, name=None):
        MergeLayer.__init__(self, [layer, layer_mask], name=name)
        self.seq_axis = seq_axis

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, inputs, **kwargs):
        att, mask = inputs

        att = T.exp(att - T.max(att, axis=self.seq_axis, keepdims=True))
        att = att * mask
        att /= T.sum(att, axis=self.seq_axis, keepdims=True)

        return att


class EfficientAttentionLayer(MergeLayer):
    """
    Layer of shape (batch_size, n_features)

    Parameters
    ----------
    attended_layer: layer of shape (batch_size, seq_length, n_features)
    attended_layer_mask: layer of shape (batch_size, seq_length, n_features)
    condition_layer: layer of shape (batch_size, n_features)
    """
    def __init__(self, attended_layer, attended_layer_mask,
                 condition_layer, gate_covariance=False, covariance_decay=None,
                 name=None):
        MergeLayer.__init__(self, [attended_layer, attended_layer_mask,
                                   condition_layer], name=name)
        self.gate_covariance = gate_covariance
        self.covariance_decay = covariance_decay
        if gate_covariance:
            n_units = attended_layer.output_shape[-1]
            self.w_gate = self.add_param(init.Constant(0.0),
                                         (n_units,), name="gate")
            self.b_gate = self.add_param(init.Constant(1.0),
                                         (1,), name="gate")
            self.b_gate = T.addbroadcast(self.b_gate, 0)

    def get_output_shape_for(self, input_shapes):
        attended_layer_shape = input_shapes[0]
        return attended_layer_shape[0], attended_layer_shape[-1]

    def get_output_for(self, inputs, **kwargs):

        # seq_input: (batch_size, seq_size, n_hidden_con)
        # seq_mask: (batch_size, seq_size)
        # condition: (batch_size, n_hidden_con)
        seq_input, seq_mask, condition = inputs

        if self.gate_covariance:
            update = T.nnet.sigmoid(
                T.sum(seq_input * self.w_gate, axis=-1, keepdims=True) +
                self.b_gate)
            seq_input *= update

        length_seq = seq_input.shape[1]
        if self.covariance_decay:
            decay = T.arange(1, length_seq+1)
            decay = (self.covariance_decay +
                     (length_seq-decay) * (1 - self.covariance_decay))
            decay = T.sqrt(decay)
            decay = decay.dimshuffle('x', 0, 'x')
            seq_input *= decay

        seq_input *= T.shape_padright(seq_mask)
        # (batch_size, n_hidden_question, n_hidden_question)
        covariance = T.batched_dot(seq_input.dimshuffle(0, 2, 1), seq_input)
        # (batch_size, n_hidden_question), equivalent to the following line:
        # att = T.sum(covariance * condition.dimshuffle((0, 'x', 1)), axis=2)
        att = 1000 * T.batched_dot(covariance, condition.dimshuffle((0, 1)))

        if not self.covariance_decay:
            att /= T.sum(seq_mask, axis=1, keepdims=True)
        # norm2_att = T.sum(att * condition, axis=1, keepdims=True)
        # att = 1000 * att / norm2_att

        return att


class CandidateOutputLayer(MergeLayer):
    """
    Layer of shape (batch_size, n_outputs)
    Parameters
    ----------
    output_layer: layer of shape (batch_size, n_outputs)
    candidate_layer: layer of shape (batch_size, max_n_candidates)
    candidate_mask_layer: layer of shape (batch_size, max_n_candidates)
    """
    def __init__(self, output_layer, candidate_layer, candidate_mask_layer,
                 non_linearity=T.nnet.softmax, name=None):
        MergeLayer.__init__(self, [output_layer, candidate_layer,
                                   candidate_mask_layer], name=name)
        self.non_linearity = non_linearity

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, inputs, **kwargs):

        # out: (batch_size, n_entities)
        # cand: (batch_size, n_candidates)
        # cand_mask: (batch_size, n_candidates)
        out, cand, cand_mask = inputs

        n_entities = self.input_shapes[0][1]
        is_candidate = T.eq(
            T.arange(n_entities, dtype='int32')[None, None, :],
            T.switch(cand_mask, cand,
                     -T.ones_like(cand))[:, :, None]).sum(axis=1)

        out = T.switch(is_candidate, out, -1000 * T.ones_like(out))

        return self.non_linearity(out)


class ForgetSizeLayer(Layer):
    def __init__(self, incoming, axis=-1, **kwargs):
        Layer.__init__(self, incoming, **kwargs)
        self.axis = axis

    def get_output_for(self, input, **kwargs):
        return input

    def get_output_shape_for(self, input_shape):
        shape = list(input_shape)
        shape[self.axis] = None
        return tuple(shape)


def apply_mask(layer_seq, layer_seq_mask):
    """
    seq: layer of shape (batch_size, length_seq, n_features)
    seq_mask: layer of shape (batch_size, length_seq)
    """
    return ElemwiseMergeLayer(
        [ForgetSizeLayer(dimshuffle(layer_seq_mask,
                                    (0, 1, 'x'))), layer_seq], T.mul)


def create_deep_rnn(layer, layer_class, depth, layer_mask=None, residual=False,
                    skip_connections=False, bidir=False, dropout=None,
                    init_state_layers=None, **kwargs):
    """
    (Deep) RNN with possible skip/residual connections, bidirectional, dropout
    """
    layers = [layer]
    for i in range(depth):
        if skip_connections and i > 0:
            layer = concat([layers[0], layer], axis=2)

        if init_state_layers:
            hid_init = init_state_layers[i]
        else:
            hid_init = init.Constant(0.)

        new_layer = layer_class(layer, hid_init=hid_init,
                                mask_input=layer_mask, **kwargs)

        if bidir:
            layer_bw = layer_class(layer, mask_input=layer_mask,
                                   backwards=True, **kwargs)
            new_layer = concat([new_layer, layer_bw], axis=2)

        if residual:
            layer = ElemwiseSumLayer([layer, new_layer])
        else:
            layer = new_layer

        if skip_connections and i == depth-1:
            layer = concat([layer] + layers[1:], axis=2)

        if dropout:
            layer = DropoutLayer(layer, p=dropout)

        layers.append(layer)

    return layers[1:]


def non_flattening_dense_layer(layer, mask, num_units, *args, **kwargs):
    """
    Lasagne dense layer which is not flattening the outputs
    """
    batchsize, seqlen = mask.input_var.shape
    l_flat = ReshapeLayer(layer, (-1, [2]))
    l_dense = DenseLayer(l_flat, num_units, *args, **kwargs)
    return ReshapeLayer(l_dense, (batchsize, seqlen, -1))
