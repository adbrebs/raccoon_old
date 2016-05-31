import numpy as np

import theano
import theano.tensor as T
from theano import shared
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

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

        buff = T.reshape(seq_or_batch, (-1,))
        self.sub = self.w[buff]
        return T.reshape(self.sub, (seq_or_batch.shape[0],
                                    seq_or_batch.shape[1], -1))


class DropoutLayer:
    def __init__(self, p):
        self.p = p
        self.srng = RandomStreams(seed=np.random.randint(10e8))

    def apply(self, x, training_time):
        if training_time:
            return x * self.srng.binomial(x.shape, p=1-self.p,
                                          dtype=theano.config.floatX) / (
                   1-self.p)

        return x