import numpy as np

import theano
from theano import shared
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from utils import create_uneven_weight, logsumexp

floatX = theano.config.floatX


class MixtureGaussians:
    def __init__(self, ls_n_in, n_mixtures, n_dim, initializer, eps=1e-5):
        if not isinstance(ls_n_in, (tuple, list)):
            ls_n_in = [ls_n_in]

        self.n_in = sum(ls_n_in)
        self.n_mixtures = n_mixtures
        self.n_dim = n_dim
        self.eps = eps

        self.n_out = (n_mixtures +  # proportions
                      n_mixtures * n_dim +  # means
                      n_mixtures * n_dim)  # stds

        w_in_mat = create_uneven_weight(ls_n_in, self.n_out, initializer)
        self.w = shared(w_in_mat, 'w_mixt')
        self.b = shared(np.random.normal(
            0, 0.001, size=(self.n_out,)).astype(floatX), 'b_mixt')

        self.params = [self.w, self.b]

    def compute_parameters(self, h, bias):
        """
        b = batch or batch*seq

        h: (batch or batch*seq, features)

        returns:
            prop: (b, n)
            mean: (b, n, d)
            std: (b, n, d)
        """
        n, d = self.n_mixtures, self.n_dim
        out = T.dot(h, self.w) + self.b
        prop = T.nnet.softmax(out[:, :n]*(1 + bias))
        mean = out[:, n:n + d*n]
        mean = T.reshape(mean, (-1, n, d))
        std = T.exp(out[:, n + d*n:n + 2*d*n] - bias) + self.eps
        std = T.reshape(std, (-1, n, d))

        return prop, mean, std

    def prediction(self, h, bias):
        srng = RandomStreams(seed=42)

        prop, mean, std = self.compute_parameters(h, bias)

        mode = T.argmax(srng.multinomial(pvals=prop, dtype=prop.dtype), axis=1)

        bs = mean.shape[0]
        v = T.arange(0, bs)
        m = mean[v, mode]  # (bs, d)
        s = std[v, mode]  # (bs, d)

        normal = srng.normal((bs, self.n_dim))  # (bs, d)
        normal_n = m + s * normal

        return normal_n

    def apply(self, h_seq, mask_seq, tg_seq):
        """
        h_seq: (seq, batch, features)
        mask_seq: (seq, batch)
        tg_seq: (seq, batch, features=63)
        """
        d = self.n_dim

        h_seq = T.reshape(h_seq, (-1, h_seq.shape[-1]))
        tg_seq = T.reshape(tg_seq, (-1, tg_seq.shape[-1]))
        mask_seq = T.reshape(mask_seq, (-1,))

        # prop: (b, n), mean: (b, n, d), std: (b, n, d)
        prop, mean, std = self.compute_parameters(h_seq, .0)

        # (b, n, d)
        tg_s = (T.shape_padaxis(tg_seq, 1) - mean) / std

        tmp = (-0.5*d*T.log(2*np.pi) - T.log(T.prod(std, axis=-1))
               - 0.5*T.sum(tg_s*tg_s, axis=-1) + T.log(prop))

        c = -logsumexp(tmp, axis=1)

        c = T.sum(c * mask_seq) / T.sum(mask_seq)
        c.name = 'negll'

        max_prop = T.argmax(prop, axis=1).mean()
        max_prop.name = 'max_prop'

        std_max_prop = T.argmax(prop, axis=1).std()
        std_max_prop.name = 'std_max_prop'

        min_std = T.min(std)
        min_std.name = 'min_std_mixture'

        return c, [c, max_prop, std_max_prop, min_std]


class SquareOutput:
    def __init__(self, ls_n_in, n_dim, initializer):
        if not isinstance(ls_n_in, (tuple, list)):
            ls_n_in = [ls_n_in]

        self.n_in = sum(ls_n_in)
        self.n_dim = n_dim

        self.n_out = n_dim

        w_in_mat = create_uneven_weight(ls_n_in, self.n_out, initializer)
        self.w = shared(w_in_mat, 'w_mixt')
        self.b = shared(np.random.normal(
            0, 0.001, size=(self.n_out,)).astype(floatX), 'b_mixt')

        self.params = [self.w, self.b]

    def compute_parameters(self, h, bias):
        """
        b = batch or batch*seq

        h: (batch or batch*seq, features)

        returns:
            prop: (b, n)
            mean: (b, n, d)
            std: (b, n, d)
        """
        return T.dot(h, self.w) + self.b

    def prediction(self, h, bias):

        # (b, n, d)
        out = self.compute_parameters(h, bias)

        return out

    def apply(self, h_seq, mask_seq, tg_seq):
        """
        h_seq: (seq, batch, features)
        mask_seq: (seq, batch)
        tg_seq: (seq, batch, features=63)
        """
        h_seq = T.reshape(h_seq, (-1, h_seq.shape[-1]))
        tg_seq = T.reshape(tg_seq, (-1, tg_seq.shape[-1]))
        mask_seq = T.reshape(mask_seq, (-1,))

        # out: (b, d)
        out = self.compute_parameters(h_seq, .0)

        buff = (out - tg_seq)
        c = (buff*buff).sum(axis=1) * mask_seq
        c = c.sum()
        c.name = 'mse'
        mse_monitoring = {'metric': c, 'counter': mask_seq.sum()}
        loss = c / mask_seq.sum()
        loss.name = 'mse'

        return loss, [mse_monitoring]
