import numpy as np

import theano
import theano.tensor as T
from theano import shared
from theano.sandbox.rng_mrg import MRG_RandomStreams as MRG_RandomStreams
from theano.tensor.shared_randomstreams import RandomStreams

from . import create_parameter


class FFLayer:
    def __init__(self, n_in, n_out, w_initializer, b_initializer,
                 non_linearity=None):
        self.n_in = n_in
        self.n_out = n_out
        if not non_linearity:
            non_linearity = lambda x: x
        self.non_linearity = non_linearity

        self.w = create_parameter(w_initializer, (n_in, n_out), 'w_ff')
        self.b = create_parameter(b_initializer, (n_out,), 'b_ff')

        self.params = [self.w, self.b]

    def apply(self, x):
        return self.non_linearity(T.dot(x, self.w) + self.b)


class EmbeddingLayer:
    def __init__(self, vocab_size, embedding_size, initializer):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        self.w = shared(initializer.sample((vocab_size, embedding_size)),
                        'w_embedding')
        self.params = [self.w]

    def apply(self, seq_or_batch):
        self.sub = self.w[seq_or_batch]
        return self.sub


class DropoutLayer:
    def __init__(self, p):
        self.p = p
        self.srng = MRG_RandomStreams(seed=np.random.randint(10e8))

        self.params = []

    def apply(self, x, training_time):
        if training_time:
            return x * self.srng.binomial(x.shape, p=1-self.p,
                                          dtype=theano.config.floatX) / (
                   1-self.p)

        return x


class MixtureGaussiansSoftmax:
    def __init__(self, n_inputs, n_outputs, means, priors, stds,
                 train_means, train_priors, train_stds, jacobian_factor):

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.means = means  # (n_out, n_in)
        self.priors = priors  # (n_out,)
        self.stds = stds  # (n_in,)
        self.jacobian_factor = jacobian_factor

        self.params = []
        if train_means:
            self.params.append(means)
        if train_priors:
            self.params.append(priors)
        if train_stds:
            self.params.append(stds)

    def compute_bw(self):

        w = self.means / T.shape_padleft(T.sqr(self.stds))

        b = T.log(self.priors) - 0.5*(self.means * w).sum(axis=1)

        # out = T.dot(input, self.w.T) + self.b
        # out = T.nnet.softmax(out)
        #
        # # (batch_size, )
        # return -T.nnet.categorical_crossentropy(out, tg)

        return b, w

    def compute_post_negll(self, h, tg):
        # input: (batch_size, n_in)

        ### Posterior p(c/x)

        # b1: (n_out, n_in)
        b1 = self.means / (T.sqr(self.stds) + 1e-8)
        # b2: (batch_size, n_out)
        b2 = - 0.5*T.sqrt((b1 * self.means).sum(axis=-1)) + T.dot(h, b1.T)

        # buff: (batch_size, n_out)
        buff = b2 + T.log(self.priors)

        post = T.nnet.softmax(buff)

        return T.nnet.categorical_crossentropy(post, tg)

    def compute_joint_loss(self, h, tg):
        # input: (batch_size, n_in)
        srng = RandomStreams(seed=234)
        a = srng.permutation(n=h.shape[0], size=(1,))[0]
        h_noised = h[a]

        means = self.means[tg]

        # z: (batch_size, n_in)
        z = (h - means) / (self.stds + 1e-6)

        l = (T.log(self.priors[tg])
             - T.log(self.stds).sum()
             - 0.5 * (z ** 2).sum(axis=-1))

        # add Jacobian
        equal = T.eq(h.sum(axis=1), h_noised.sum(axis=1))
        inc = T.switch(equal, T.zeros_like(equal, equal.dtype),
                       self.jacobian_factor*0.5*self.n_inputs * (
                           T.log(1e-8 + ((h - h_noised)**2).sum(axis=-1))))
        l += inc

        return -l



