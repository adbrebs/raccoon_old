import numpy as np

import theano
from theano import tensor

from extensions import Extension

class AdaptZLoss(Extension):
    def __init__(self, freq, activation, input, target_idx, task_loss, surrogate_loss,
                 hyperparameter, learning_rate, batch_generator, n_batches,
                 factor=1.5, n_updates=10):
        Extension.__init__(self, 'adapt_zloss', freq)

        self.batch_generator = batch_generator
        self.n_batches = n_batches
        self.learning_rate = learning_rate
        self.hyperparameter = hyperparameter
        self.factor = factor
        self.n_updates = n_updates

        # grad = theano.grad(surrogate_loss, activation)
        # new_activation = activation - learning_rate * grad
        self.fun_activation = theano.function([input], activation)

        activation_bis = tensor.matrix()
        surr_loss_bis = theano.clone(surrogate_loss,
                                     replace={activation: activation_bis})
        grad = theano.grad(surr_loss_bis, activation_bis)
        new_activation = activation_bis - 100*learning_rate * grad

        task_loss_bis = theano.clone(task_loss,
                                     replace={activation: new_activation})

        self.fun_update_task_loss = theano.function(
                [activation_bis, target_idx], [task_loss_bis, new_activation])
        # self.fun_update_task_loss = theano.function(
        #         [activation, target_idx],
        #         theano.clone(task_loss, replace={activation: new_activation}))

    # def evaluate_hyper_param(self, param):
    #     self.hyperparameter.set_value(param)
    #
    #     c = 0
    #     score = 0
    #     for batch_input, target_input in self.batch_generator():
    #         if c == self.n_batches:
    #             break
    #         activation = self.fun_activation(batch_input)
    #         for i in range(self.n_updates):
    #             s, activation = self.fun_update_task_loss(activation, target_input)
    #         score += s
    #         c += 1
    #     return score / c

    def execute_virtual(self, batch_id):

        init_param_value = self.hyperparameter.get_value()

        factor = np.random.uniform(0, self.factor)
        param_values = np.array([init_param_value,
                                 init_param_value * factor,
                                 init_param_value * factor**2,
                                 init_param_value / factor,
                                 init_param_value / (factor**2)],
                                dtype='float32')

        scores = np.zeros_like(param_values)

        c = 0
        for batch_input, target_input in self.batch_generator():
            if c == self.n_batches:
                break
            activation = self.fun_activation(batch_input)
            for i in range(len(param_values)):
                self.hyperparameter.set_value(param_values[i])
                new_activation = activation
                for j in range(self.n_updates):
                    s, new_activation = self.fun_update_task_loss(new_activation, target_input)
                scores[i] += s
            c += 1
        scores /= c

        print param_values
        print scores
        best_param_value = param_values[np.argmin(scores)]

        msg = 'hyper_param equals {}'.format(best_param_value)
        if best_param_value == init_param_value:
            msg += ' (unchanged)'
        self.hyperparameter.set_value(best_param_value)

        return [msg]
