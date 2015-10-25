import sys
import os
import time
import cPickle
import heapq
import numpy as np
import theano

from utils import print_wrap, remove_duplicates

floatX = theano.config.floatX


class MonitoredQuantity:
    """
    This class allows to monitor quantities that are not theano tensors.
    These quantities can be calculated from computed tensor values returned by
    the trainer object. These theano tensors have to be indicated in
    required_tensors.

    If there are multiple quantities computed, name has to be a list with as
    many names as quantities returned by the calculate.

    See KErrorRate for an example.

    Parameters:
    -----------
    name_or_names: str or list of strings
        names of the quantities being computed
    required_tensors: list of Theano tensors (default=None)
        if specified, the values of these tensors will be provided to the
    """
    def __init__(self, name_or_names, required_tensors=None):
        self.name = name_or_names
        if not required_tensors:
            required_tensors = []
        self.required_tensors = required_tensors

    def calculate(self, *inputs):
        """
        inputs is a list of the computed tensor values returned by the trainer.
        The order of the inputs is the SAME as the order provided in
        required_tensors.

        Returns either a single value of a list of values
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
    def __init__(self, idx_target, output, k):
        name_or_names = '{} error rate'.format(k)
        MonitoredQuantity.__init__(self, name_or_names, [idx_target, output])
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
    __init__ and execute.

    It has to be registered to a Trainer object, either as an extension or as
    and ending condition.

    Parameters
    ----------
    name_extension : string
        The name of the extension
    freq : int
        The frequency with which the extension is called
    apply_at_the_end: bool, default False
        Apply the extension at the end of training or when training is
        interrupted
    """
    def __init__(self, name_extension, freq, apply_at_the_end=False):
        self.name_extension = name_extension
        self.freq = freq
        self.apply_at_the_end = apply_at_the_end

    def check(self, batch_id):
        """
        This method is called by the trainer object at every batch during
        training.

        Returns
        -------
        list of strings or None
            None should be returned when the extension is not executed.
        """
        if not batch_id % self.freq:
            return self.execute(batch_id)
        return None

    def execute(self, batch_id):
        """The method that is called when the extension is executed.

        Returns
        -------
        list of strings
            The lines to be printed during training. They will automatically be
            indented.
        """
        return ['Extension was executed']


class EndCondition(object):
    """Class responsible for terminating the training.

    It has to be registered to a Trainer object.

    check_condition_virtual is the only method to be re-implemented if you
    inherit from this class. Check MaxIteration for a simple exemple.

    Parameters:
    -----------
    name: string
        Name of the ending condition.
    freq: int
        Frequency to which this ending condition should be checked.
    """
    def __init__(self, name, freq):
        self.name = name
        self.freq = freq

    def check_condition(self, batch_id):
        if not batch_id % self.freq:
            return self.check_condition_virtual(batch_id)
        return False

    def check_condition_virtual(self, batch_id):
        """
        Method that checks if the ending condition is met.
        If it is met, it should return a list of strings, one string per line
        to be printed.
        If it is not met, it should return False.
        """
        return False


class MaxIteration(EndCondition):
    """Stops training when a maximal number of iterations is reached.
    """
    def __init__(self, freq, max_batchs):
        EndCondition.__init__(self, 'Max Iteration', freq)
        self.max_batchs = max_batchs

    def check_condition_virtual(self, batch_id):
        if batch_id > self.max_batchs:
            return ['Maximal number of batches reached']
        return False


class MaxTime(EndCondition):
    """Stops training when a certain amount of training time is reached
    """
    def __init__(self, freq, max_time=3600*48):
        EndCondition.__init__(self, 'Max Iteration', freq)
        self.max_time = max_time
        self.begin_time = time.clock()

    def check_condition_virtual(self, batch_id):
        if (time.clock() - self.begin_time) > self.max_time:
            return ['Time exceeded']
        return False


class ExtVarMonitor(Extension):
    """Extension to monitor MonitoredQuantity objects that don't depend
    on theano tensors.

    Parameters:
    -----------
    monitored_quantities: list of MonitoredQuantity objets
        the list of quantities to be monitored. These quantities should not
        require any theano tensor values to be computed.
    """
    def __init__(self, name_extension, freq, monitored_quantities):
        Extension.__init__(self, name_extension, freq)
        self.mon_quantities = monitored_quantities

        # Stores the current values of the monitored quantities
        self.current_values = None
        self.current_spent_time = None

        # Stores all the values of the monitored quantities.
        self.history = []
        self.iterations = []

    def execute(self, batch_id):
        self.current_spent_time = np.zeros(len(self.mon_quantities))

        # Compute the quantities from the tensor values
        quantities = []
        for i, quantity in enumerate(self.mon_quantities):
            begin = time.clock()
            res = quantity.calculate()
            self.current_spent_time[i] = time.clock() - begin
            if not isinstance(res, list):
                res = [res]
            quantities.extend(res)

        self.current_values = np.array(quantities)

        strs = self.get_str()
        self.history.append(self.current_values)
        self.iterations.append(batch_id)
        return strs

    def get_str(self):
        strs = []
        c = 0
        for i, var in enumerate(self.mon_quantities):
            if isinstance(var.name, list):
                strs.append('Computed together in {:.3g} seconds:'.format(
                    self.current_spent_time[i]))
                for name in var.name:
                    strs.append('  ' + name + ': {}'.format(
                        self.current_values[c]))
                    c += 1
            else:
                strs.append(var.name + ': [{:.3g} seconds] {}'.format(
                    self.current_spent_time[i], self.current_values[c]))
                c += 1

        return strs


class VarMonitor(Extension):
    """Extension to monitor theano tensors or MonitoredQuantity objects that
    depend on theano tensor values.

    It compiles a theano function internally.

    This is an abstract class.

    Parameters
    ----------
    inputs: list of theano tensors
        tensors necessary to compute the monitored_variables
    monitored_variables : list of theano tensors or MonitoredQuantity objects
        the list of variables that are monitored by this extension
    updates: list of theano updates, optional, default=None
        Updates fo be performed by the theano function.
    """
    def __init__(self, name_extension, freq, inputs, monitored_variables,
                 updates=None):
        Extension.__init__(self, name_extension, freq)
        self.mon_variables = monitored_variables

        # Tensors that have to be computed.
        self.required_tensors = []  # initialization

        # Split tensors and quantities. Record all the tensors to be computed
        def add_tensor(tensor, a_list):
            try:
                idx = self.required_tensors.index(tensor)
                a_list.append(idx)
            except ValueError:
                self.required_tensors.append(tensor)
                a_list.append(len(self.required_tensors)-1)

        self.links = {}
        for var in monitored_variables:
            links = []
            if isinstance(var, MonitoredQuantity):
                for t in var.required_tensors:
                    add_tensor(t, links)
            elif isinstance(var, theano.Variable):
                add_tensor(var, links)
            else:
                raise ValueError('monitored_variables should contain either '
                                 'theano tensors or MonitoredQuantity objects')
            self.links[var] = links

        # Function that will output the values of the required tensors for
        # given inputs.
        self.f = theano.function(inputs, self.required_tensors, updates=updates)

        # Stores the current values of the monitored variables
        self.current_values = np.zeros(len(self.required_tensors), dtype=floatX)
        self.current_spent_time = 0

        # Stores all the values of the monitored variables.
        self.history = []
        self.iterations = []

    def execute(self, batch_id):
        self.compute_current_values()
        strs = self.get_str()
        self.history.append(self.current_values)
        self.iterations.append(batch_id)
        # Reset current_values to zero for the next raccoon cycle
        self.current_values = np.zeros_like(self.current_values)
        return strs

    def compute_current_values(self):
        """Computes the current values of the monitored variables. Called by
        the execute method.
        """
        pass

    def inc_values(self, *inputs):
        """Computes the values for the monitored variables for given inputs for
        a given batch.
        """
        begin = time.clock()

        # List of values of the required tensors. We have to compute the
        # quantities from them.
        tensor_values = self.f(*inputs)

        # Compute the quantities from the tensor values
        var_values = []
        for var in self.mon_variables:
            values = [tensor_values[i] for i in self.links[var]]
            if isinstance(var, theano.Variable):
                res = values[0]
            else:
                res = var.calculate(*values)
            if not isinstance(res, list):
                res = [res]
            var_values.extend(res)

        self.current_values += np.array(var_values)

        self.current_spent_time += (time.clock() - begin)

    def get_str(self):
        strs = ['timing: {:.3g} seconds spent for 1000 batches'.format(
            1000 * self.current_spent_time)]
        c = 0
        for var in self.mon_variables:
            if isinstance(var.name, list):
                for name in var.name:
                    strs.append(name + ': {}'.format(self.current_values[c]))
                    c += 1
            else:
                strs.append(var.name + ': {}'.format(self.current_values[c]))
                c += 1

        return strs


class ValMonitor(VarMonitor):
    """
    Extension to monitor tensor variables on an external fuel stream.
    """
    def __init__(self, name_extension, freq, inputs, monitored_variables,
                 stream):
        VarMonitor.__init__(self, name_extension, freq, inputs,
                            monitored_variables)
        self.stream = stream

    def compute_current_values(self):
        iterator = self.stream.get_epoch_iterator()
        c = 0.0
        for batch_input, target_input in iterator:
            self.inc_values(batch_input, target_input[:, 0])
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


class LearningRateDecay(Extension, EndCondition):
    """
    If does not improve for absolute_patience, then the training stops
    """
    def __init__(self, val_monitor, var_name, learning_rate, patience=5,
                 max_patience=7, decay_rate=2., min_value=1e-12):
        Extension.__init__(self, 'Learning rate', val_monitor.freq)
        EndCondition.__init__(self, 'Learning rate', val_monitor.freq)
        self.var_name = var_name
        self.lr = learning_rate
        self.patience = patience
        self.absolute_patience = max_patience
        self.decay_rate = decay_rate
        self.waiting = 0
        self.absolute_waiting = 0
        self.val_monitor = val_monitor
        self.min_value = min_value

        # Determine index of var_name in val_monitor
        for i, var in enumerate(val_monitor.mon_variables):
            if var.name == var_name:
                self.var_idx = i
                break

        self.best_value = np.inf

    def execute(self, batch_id):
        current_value = self.val_monitor.history[-1][self.var_idx]
        if current_value < self.best_value:
            self.best_value = current_value
            self.waiting = 0
            self.absolute_waiting = 0
        else:
            self.waiting += 1
            self.absolute_waiting += 1
        strs = ['Learning rate: {}, waiting {}/{}'.format(
            self.lr.get_value(), self.waiting, self.patience)]
        if self.waiting > self.patience:
            self.lr.set_value(self.lr.get_value()/self.decay_rate)
            self.waiting = 0
            strs.append('Learning rate decreased')
        return strs

    def check_condition_virtual(self, batch_id):
        current_value = self.val_monitor.history[-1][self.var_idx]

        if self.absolute_waiting > self.absolute_patience:
            return ['Patience exceeded']
        if current_value < self.min_value:
            return ['Learning rate too small']
        return False


class Saver(Extension):
    """Extension to pickle objects.

    Only the compute_object method should be re-implemented.
    """
    def __init__(self, name_extension, freq, folder_path, file_name):
        super(Saver, self).__init__(name_extension, freq, apply_at_the_end=True)
        self.folder_path = folder_path
        self.file_name = file_name

    def execute(self, batch_id):
        file_handle = open(os.path.join(
            self.folder_path, self.file_name + '.pkl'), 'wb')
        obj, msg = self.compute_object()
        cPickle.dump(obj, file_handle)
        return msg

    def compute_object(self):
        """
        It should return a tuple of to elements. The first is the object to be
        saved. The second element is a list of strings, each string
        representing a line to be printed when the extension is executed.
        """
        return None, None


class NetworkSaver(Saver):
    """Saves the parameters of the network.
    """
    def __init__(self, net, freq, folder_path, file_name='net.pkl'):
        super(NetworkSaver, self).__init__('Network Saver', freq,
                                           folder_path, file_name)
        self.net = net

    def compute_object(self):
        return (self.net.get_param_values(),
                ['Network dumped into {}'.format(self.folder_path)])


class VariableSaver(Saver):
    """Saves the history of a ValMonitor extension
    """
    def __init__(self, var_monitor, freq, folder_path, file_name=None):
        if not file_name:
            file_name = var_monitor.name_extension
        super(VariableSaver, self).__init__('Variables saver', freq,
                                            folder_path, file_name)
        self.var_monitor = var_monitor

    def compute_object(self):
        np_history = np.array(self.var_monitor.history)
        np_iterations = np.array(self.var_monitor.iterations)
        return ((np_iterations, np_history),
                ['Variable histories dumped into {}'.format(self.folder_path)])


class Trainer:
    def __init__(self, train_monitor, extensions, end_conditions):
        self.train_monitor = train_monitor
        self.extensions = [train_monitor] + extensions
        self.end_conditions = end_conditions

    def print_extensions_logs(self, extensions_logs):
        for i, logs in enumerate(extensions_logs):
            # If extension has not ran, we don't print anything
            if not logs:
                continue

            print print_wrap(
                '{}:'.format(self.extensions[i].name_extension), 1)
            for line in logs:
                print print_wrap(line, 2)

    def print_end_conditions_logs(self, cond_logs):
        for i, logs in enumerate(cond_logs):
            # If extension has not ran, we don't print anything
            if not logs:
                continue

            print print_wrap(
                '{}:'.format(self.end_conditions[i].name), 1)
            for line in logs:
                print print_wrap(line, 2)

    def apply(self, epoch, iteration, *inputs):
        self.train_monitor.train(*inputs)

        if iteration == 0:
            return False

        extensions_logs = [ext.check(iteration) for ext in self.extensions]
        cond_logs = [cond.check_condition(iteration)
                     for cond in self.end_conditions]

        # If no extensions are active
        if not any(extensions_logs) and not any(cond_logs):
            return False

        print 'Epoch {}, iteration {}:'.format(epoch, iteration)
        self.print_extensions_logs(extensions_logs)
        self.print_end_conditions_logs(cond_logs)
        print '-'*79
        sys.stdout.flush()  # Important if output is redirected

        if any(cond_logs):
            return True
        return False

    def finish(self, batch_id):
        print 'Training finished'
        print 'Computing extensions...',
        print 'Done!'
        extensions_logs = [ext.execute(batch_id)
                           for ext in self.extensions if ext.apply_at_the_end]
        self.print_extensions_logs(extensions_logs)
