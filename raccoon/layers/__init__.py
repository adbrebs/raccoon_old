import theano
from theano import shared
import theano.tensor as T

import numpy as np

from utils import create_uneven_weight

theano.config.floatX = 'float32'
floatX = theano.config.floatX


def normal_mat(size):
    return np.random.normal(0, 0.001, size=size).astype(floatX)


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


class FFLayer:
    def __init__(self, n_in, n_out, initializer,
                 non_linearity=None):
        self.n_in = n_in
        self.n_out = n_out
        if not non_linearity:
            non_linearity = lambda x: x
        self.non_linearity = non_linearity

        self.w = shared(initializer.sample((n_in, n_out)), 'w_ff')
        self.b = shared(np.random.normal(0, 0.0001, (n_out,)).astype('float32'), 'b_ff')

        self.params = [self.w, self.b]

    def apply(self, x):
        return self.non_linearity(T.dot(x, self.w) + self.b)
