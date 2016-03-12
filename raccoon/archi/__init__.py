import theano
from theano import shared
import theano.tensor as T
from theano.gradient import grad_clip

import numpy as np

theano.config.floatX = 'float32'
floatX = theano.config.floatX


def normal_mat(size):
    return np.random.normal(0, 0.01, size=size).astype(floatX)


class GRULayer:
    def __init__(self, n_in, n_out, initializer):
        self.w = shared(initializer.sample((n_in, n_out)), 'w_gru')
        self.u = shared(initializer.sample((n_out, n_out)), 'u_gru')
        self.b = shared(normal_mat((n_out,)), 'b_gru')

        self.wr = shared(initializer.sample((n_in, n_out)), 'wr_gru')
        self.ur = shared(initializer.sample((n_out, n_out)), 'ur_gru')
        self.br = shared(normal_mat((n_out,)), 'br_gru')

        self.wu = shared(initializer.sample((n_in, n_out)), 'wu_gru')
        self.uu = shared(initializer.sample((n_out, n_out)), 'uu_gru')
        self.bu = shared(normal_mat((n_out,)), 'bu_gru')

        self.params_scan = [self.u, self.ur, self.uu]

        self.params = ([self.w, self.wr, self.wu, self.b, self.br, self.bu] +
                       self.params_scan)

    def apply(self, seq_coord, seq_mask, h_ini):

        seq_x_in = T.dot(seq_coord, self.w) + self.b
        seq_x_r = T.dot(seq_coord, self.wr) + self.br
        seq_x_u = T.dot(seq_coord, self.wu) + self.bu

        def gru_step(x_in, x_r, x_u, mask, h_pre, u, ur, uu):

            r_gate = T.nnet.sigmoid(x_r + T.dot(h_pre, ur))
            u_gate = T.nnet.sigmoid(x_u + T.dot(h_pre, uu))

            h_new = T.tanh(x_in + T.dot(r_gate * h_pre, u))

            h = (1-u_gate)*h_pre + u_gate*h_new

            h = mask[:, None]*h + (1-mask[:, None])*h_pre
            # h = grad_clip(h, -10, 10)

            return h

        seq_h, scan_updates = theano.scan(
                fn=gru_step,
                sequences=[seq_x_in, seq_x_r, seq_x_u, seq_mask],
                outputs_info=[h_ini],
                non_sequences=self.params_scan,  # les poids utilises
                strict=True)

        return seq_h, scan_updates
