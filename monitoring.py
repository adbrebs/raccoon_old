import os
import time
import cPickle

import theano
floatX = theano.config.floatX

import heapq

import numpy as np

from nlm.utilities import print_wrap, remove_duplicates


class MonitoredQuantity:
    """
    This class allows to monitor quantities that are not theano tensors.
    These quantities are computed from computed tensor values returned by the
    trainer object. These theano tensors have to be precised in
    required_tensors.
    """
    def __init__(self, name, required_tensors):
        self.name = name
        self.required_tensors = required_tensors

    def calculate(self, *inputs):
        """
        inputs is a of the computed tensor values returned by the trainer.
        The order of the inputs is the SAME as the order provided in
        required_tensors.
        """
        pass


class KErrorRate(MonitoredQuantity):
    """
    Computes the k error rate for a classification model.
    Parameters
    ----------
    idx_target: theano 1D tensor of shape (batch_size,)
        the targets of a batch
    output: theano 2D tensor of shape (batch_size, output_size)
        the outputs of a batch
    """
    def __init__(self, name, idx_target, output, k):
        MonitoredQuantity.__init__(self, name, [idx_target, output])
        self.k = k

    def calculate(self, *inputs):
        idx_target, output = inputs

        bs, out_dim = output.shape

        err_rate = 0
        for i in range(bs):
            idxs = heapq.nlargest(self.k, range(out_dim), output[i].take)
            if idx_target[i] not in idxs:
                err_rate += 1.0

        return err_rate / bs


class Extension(object):
    """
    Extensions are objects regularly called during the training process.
    More precisely, their check method will be called every freq batches.
    If you inherit from Extension, the only method you should implement are
    __init__ and apply.

    Parameters
    ----------
    name_extension : string
        The name of the extension
    freq : int
        The frequency with which the extension is called
    """
    def __init__(self, name_extension, freq):
        self.name_extension = name_extension
        self.freq = freq

    def check(self, batch_id):
        """
        This method is called by the trainer object at every batch during
        training.
        """
        if not batch_id % self.freq:
            return self.apply()
        return None

    def apply(self):
        """The method that is called when the extension runs.

        Returns
        -------
        list of strings
            The lines to be printed during training. You should not care about
            indentation
        """
        return ['Extension was executed']


class VarMonitor(Extension):
    """Extension to monitor tensor variables or MonitoredQuantity objects.

    It compiles a theano function.

    Parameters
    ----------
    monitored_variables : list of theano tensors or MonitoredQuantity objects
        the list of variables that are monitored by this extension
    """
    def __init__(self, name_extension, freq, inputs, monitored_variables,
                 updates=None):
        Extension.__init__(self, name_extension, freq)
        self.mon_variables = monitored_variables

        # tensors that are monitored
        self.mon_tensors = []

        # MonitoredQuantity objects that are monitored
        self.mon_quantities = []

        # Tensors that are computed. Includes mon_tensors at the beginning of
        # the list and the tensors required by the mon_quantities at the end.
        self.required_tensors = [self.mon_tensors]

        # Split tensors and quantities. Record all the tensors to be computed
        for var in monitored_variables:
            if isinstance(var, MonitoredQuantity):
                self.mon_quantities.append(var)
                self.required_tensors.extend(var.required_tensors)
            elif isinstance(var, theano.Variable):
                self.mon_tensors.append(var)
            else:
                raise ValueError('monitored_variables should contain either '
                                 'theano tensors or MonitoredQuantity objects')

        # If two quantities require the same tensors, there might be duplicates
        self.required_tensors = remove_duplicates(self.required_tensors)

        # For quantities, link them to the tensors they require
        # quantities_links is a dictionary {quantity: links}. links is a list
        # containing the indices of the theano tensors rquired by quantity
        # (parameter required_tensors of MonitoredQuantity).
        self.quantities_links = {}
        for quantity in self.mon_quantities:
            links = []
            for t in quantity.required_tensors:
                links.append(self.required_tensors.index(t))
            self.quantities_links[quantity] = links

        # Function that will output the values of the required tensors for
        # given inputs.
        self.f = theano.function(inputs, self.required_tensors, updates=updates)

        # Stores the current values of the monitored variables
        self.current_values = np.zeros(len(monitored_variables), dtype=floatX)
        self.current_spent_time = 0

        # Stores all the values of the monitored variables.
        self.history = []

    def apply(self):
        self.compute_current_values()
        strs = self.get_str()
        self.history.append(self.current_values)
        # Reset current_values to zero for the next raccoon cycle
        self.current_values = np.zeros_like(self.current_values)
        return strs

    def compute_current_values(self):
        """Computes the current values of the monitored variables. See apply
        """
        pass

    def inc_values(self, *inputs):
        """Computes the values for the monitored variables for given inputs for
        a given batch.
        """
        begin = time.clock()

        # List of values of the required tensors. We have to compute the
        # quantities from them.
        list_values = self.f(*inputs)

        # Compute the quantities from the tensor values
        quantities = []
        for quantity in self.mon_quantities:
            quantities.append(quantity.calculate(
                *[list_values[i] for i in self.quantities_links[quantity]]))

        self.current_values += np.array(
            list_values[:len(self.mon_tensors)] + quantities)

        self.current_spent_time += (time.clock() - begin)

    def get_str(self):
        strs = ['timing: {:.3g} seconds spent for 1000 batches'.format(
            1000 * self.current_spent_time)]
        for val, var in zip(list(self.current_values), self.mon_variables):
            strs.append(var.name + ': {}'.format(val))
        return strs


class ValMonitor(VarMonitor):
    """
    Extension to monitor tensor variables on an external stream.
    """
    def __init__(self, name_extension, freq, inputs, monitored_variables,
                 stream):
        VarMonitor.__init__(self, name_extension, freq, inputs,
                            monitored_variables)
        self.stream = stream

    def compute_current_values(self):
        iterator = self.stream.get_epoch_iterator()
        c = 0.0
        for inputs in iterator:
            self.inc_values(*inputs)
            c += 1

        self.current_values /= c
        self.current_spent_time /= c


class TrainMonitor(VarMonitor):
    def __init__(self, freq, inputs, monitored_variables, updates):
        VarMonitor.__init__(self, 'Training', freq, inputs,
                            monitored_variables, updates)

    def compute_current_values(self):
        self.current_values /= self.freq
        self.current_spent_time /= self.freq

    def train(self, *inputs):
        self.inc_values(*inputs)


class LearningRateDecay(Extension):
    def __init__(self, val_monitor, var_name, learning_rate, patience=5,
                 decay_rate=2.):
        Extension.__init__(self, 'Learning rate', val_monitor.freq)
        self.var_name = var_name
        self.lr = learning_rate
        self.patience = patience
        self.decay_rate = decay_rate
        self.waiting = 0
        self.val_monitor = val_monitor

        # Determine index of var_name in val_monitor
        for i, var in enumerate(val_monitor.mon_variables):
            if var.name == var_name:
                self.var_idx = i
                break

        self.best_value = np.inf

    def apply(self):
        current_value = self.val_monitor.history[-1][self.var_idx]
        if current_value < self.best_value:
            self.best_value = current_value
            self.waiting = 0
        else:
            self.waiting += 1
        strs = ['Learning rate: {}, waiting {}/{}'.format(
            self.lr.get_value(), self.waiting, self.patience)]
        if self.waiting > self.patience:
            self.lr.set_value(self.lr.get_value()/self.decay_rate)
            self.waiting = 0
            strs.append('Learning rate decreased')
        return strs



class Saver(Extension):
    def __init__(self, name_extension, freq, folder_path, file_name):
        super(Saver, self).__init__(name_extension, freq)
        self.folder_path = folder_path
        self.file_name = file_name

    def apply(self):
        file = open(os.path.join(self.folder_path, self.file_name), 'wb')
        object, msg = self.compute_object()
        cPickle.dump(object, file)
        return msg

    def compute_object(self):
        return None, None


class NetworkSaver(Saver):
    def __init__(self, net, freq, folder_path, file_name='net.pkl'):
        super(NetworkSaver, self).__init__('Network Saver', freq,
                                           folder_path, file_name)
        self.net = net

    def compute_object(self):
        return (self.net.get_param_values(),
                'Network dumped into {}'.format(self.folder_path))


class VariableSaver(Saver):
    def __init__(self, var_monitor, freq, folder_path, file_name=None):
        if not file_name:
            file_name = var_monitor.name_extension
        super(VariableSaver, self).__init__('Variables saver', freq,
                                            folder_path, file_name)
        self.var_monitor = var_monitor

    def compute_object(self):
        np_history = np.array(self.var_monitor.history)
        return (np_history,
                'Variable histories dumped into {}'.format(self.folder_path))


class Trainer:
    def __init__(self, train_monitor, extensions):
        self.train_monitor = train_monitor
        self.extensions = [train_monitor] + extensions

    def apply(self, epoch, iteration, *inputs):
        self.train_monitor.train(*inputs)

        if iteration == 0:
            return

        extensions_logs = [ext.check(iteration) for ext in self.extensions]

        # If no extensions are active
        if not any(extensions_logs):
            return

        print 'Epoch {}, iteration {}:'.format(epoch, iteration)

        for i, logs in enumerate(extensions_logs):
            # If extension has not ran, we don't print anything
            if not logs:
                break

            print print_wrap(
                '{}:'.format(self.extensions[i].name_extension), 1)
            for line in logs:
                print print_wrap(line, 2)
        print '-'*79
