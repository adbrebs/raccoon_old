import os
import time
import cPickle
import numpy as np
import theano
from lasagne.layers import get_all_param_values, set_all_param_values

from quantities import MonitoredQuantity

floatX = theano.config.floatX


class Extension(object):
    """ Abstract class
    Extensions are objects regularly called during the training process.
    More precisely, their check method will be called every freq batches.
    If you inherit from Extension, the only methods you should implement are
    __init__ and execute.

    It has to be registered to a :class:`Trainer` object.

    Parameters
    ----------
    name_extension: string
        The name of the extension
    freq: int
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
        This method is called by the :class:`Trainer` object at every batch
        during training.

        Returns
        -------
        list of strings or None
            None should be returned when the extension is not executed.
        """
        if self.freq and not batch_id % self.freq:
            return True
        return False

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
    """Abstract class responsible for terminating the training.

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


class Monitor(Extension):
    """
    Base class for monitoring different types of variables:
    - external variables that do not depend on theano tensors: you should use
        :class:`ExternalVarMonitor`.
    - variables that depend directly on theano tensors computed during training:
        you should use :class:`TrainMonitor`.
    - variables that depend on theano tensors computed on another dataset or
        stream: you should use :class:`ValMonitor`.
    """
    def __init__(self, name_extension, freq, monitored_var, **kwargs):
        Extension.__init__(self, name_extension, freq, **kwargs)
        self.monitored_var = monitored_var

    def execute(self, batch_id):
        raise NotImplementedError

    def get_str(self):
        raise NotImplementedError


class ExternalVarMonitor(Monitor):
    """Extension to monitor MonitoredQuantity objects that don't depend
    on theano tensors.

    Parameters:
    -----------
    monitored_quantities: list of MonitoredQuantity objets
        the list of quantities to be monitored. These quantities should not
        require any theano tensor values to be computed.
    """
    def __init__(self, name_extension, freq, monitored_var, **kwargs):
        Monitor.__init__(self, name_extension, freq, monitored_var, **kwargs)

        # Stores the current values of the monitored quantities
        self.current_values = None
        self.current_spent_time = None

        # Stores all the values of the monitored quantities.
        self.history = []
        self.iterations = []

    def execute(self, batch_id):
        self.current_spent_time = np.zeros(len(self.monitored_var))

        # Compute the quantities from the tensor values
        quantities = []
        for i, quantity in enumerate(self.monitored_var):
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
        for timing, var in zip(self.current_spent_time, self.monitored_var):
            strs.append('Computed in {:.3g} seconds:'.format(timing))
            var.write_str(strs, self.current_values[c:c+var.n_outputs], '  ')
            c += var.n_outputs

        return strs


class VarMonitor(Monitor):
    """Extension to monitor theano tensors or :class:`MonitoredQuantity`
    objects.

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
                 updates=None, **kwargs):
        Monitor.__init__(self, name_extension, freq, monitored_variables,
                         **kwargs)

        # Total number of outputs (some monitored variables may have several
        # outputs). output_links {var: indices of outputs} links each variable
        # to the indices of its corresponding outputs
        self.n_outputs = 0
        self.output_links = {}
        for var in monitored_variables:
            if isinstance(var, MonitoredQuantity):
                n = var.n_outputs
            elif isinstance(var, theano.Variable):
                n = 1
            self.output_links[var] = range(self.n_outputs, self.n_outputs + n)
            self.n_outputs += n

        # Tensors that have to be computed.
        self.required_tensors = []  # initialization

        def add_tensor(tensor, a_list):
            """Adds a tensor to self.required_tensors if it is not already
            there. It also adds to the index of tensor in self.required_tensors
            to the provided list a_list.
            """
            try:
                idx = self.required_tensors.index(tensor)
                a_list.append(idx)
            except ValueError:
                self.required_tensors.append(tensor)
                a_list.append(len(self.required_tensors)-1)

        # input_links is a dictionary {var: list of indices}. Each element
        # contains the indices of the tensors required to compute var from from
        # required_tensors
        self.input_links = {}
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
            self.input_links[var] = links

        # Function that will output the values of the required tensors for
        # given inputs.
        self.f = theano.function(inputs, self.required_tensors, updates=updates)

        # Containers to store the current values of the monitored variables and
        # the computational time.
        self.current_values = np.zeros(self.n_outputs, dtype=floatX)
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
        self.current_spent_time = 0
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
        for var in self.monitored_var:
            values = [tensor_values[i] for i in self.input_links[var]]
            if isinstance(var, theano.Variable):
                res = values[0]
            else:
                res = var.calculate(*values)
            if not isinstance(res, (list, tuple)):
                res = [res]
            var_values.extend(res)

        self.current_values += np.array(var_values)
        self.current_spent_time += (time.clock() - begin)

    def get_str(self):
        strs = ['timing: {:.3g} seconds'.format(self.current_spent_time)]
        c = 0
        for var in self.monitored_var:
            if isinstance(var, MonitoredQuantity):
                var.write_str(strs, self.current_values[c:c+var.n_outputs])
                c += var.n_outputs
            else:
                strs.append(var.name + ': {}'.format(self.current_values[c]))
                c += 1

        return strs


class ValMonitor(VarMonitor):
    """
    Extension to monitor tensor variables and MonitoredQuantity objects on an
    external fuel stream.
    """
    def __init__(self, name_extension, freq, inputs, monitored_variables,
                 stream, **kwargs):
        VarMonitor.__init__(self, name_extension, freq, inputs,
                            monitored_variables, **kwargs)
        self.stream = stream

    def compute_current_values(self):
        c = 0.0
        for batch_input, target_input in self.stream():
            self.inc_values(batch_input, target_input)
            c += 1

        self.current_values /= c


class TrainMonitor(VarMonitor):
    """
    Extension required by `class:Trainer` to process_batch updates and monitor
    tensor variables and MonitoredQuantity objects.
    """
    def __init__(self, freq, inputs, monitored_variables, updates, **kwargs):
        VarMonitor.__init__(self, 'Training', freq, inputs,
                            monitored_variables, updates, **kwargs)

    def compute_current_values(self):
        self.current_values /= self.freq

    def train(self, *inputs):
        self.inc_values(*inputs)


class LearningRateDecay(Extension, EndCondition):
    """
    Both extension and ending condition that decreases the learning rate if
    there is no improvement on a variable monitored by a monitoring extension.
    If does not improve for absolute_patience, then the training stops.

    The frequence of this extension is the same as the :class:`Monitor` monitor
    parameter.

    Parameters:
    -----------
    monitor: :class:`Monitor` object
        :class:`Monitor` object which computes the variable you are
        interested in.
    var: MonitoredQuantity or tensor variable
        the MonitoredQuantity or tensor variable of :class:`Monitor` that you
        are interested in.
    idx: int, default=0
        if var computes several outputs, this index selects a single one.
    learning_rate: theano shared variable or float
        the variable storing the current learning rate
    patience: int, default=5
        the number of times we allow the variable to not improve before
        decreasing the learning rate
    max_patience: int, default=7
        the number of times we allow the variables to not improve before we
        stop the training.
    decay_rate: float, default=2.0
        the rate at which the learning rate is decreased
    min_value: float
        the minimal value that we tolerate for the learning rate. Below it, we
        stop training.
    network: lasagne layer
        if you want the best network to be saved.
    """
    def __init__(self, monitor, var, learning_rate, idx=0, patience=5,
                 max_patience=7, decay_rate=2., min_value=1e-12, network=None):
        Extension.__init__(self, 'Learning rate', monitor.freq)
        EndCondition.__init__(self, 'Learning rate', monitor.freq)
        self.var = var
        if not isinstance(learning_rate, theano.Variable):
            learning_rate = theano.shared(learning_rate, 'learning_rate')
        self.lr = learning_rate
        self.patience = patience
        self.absolute_patience = max_patience
        self.decay_rate = decay_rate
        self.waiting = 0
        self.absolute_waiting = 0
        self.val_monitor = monitor
        self.min_value = min_value

        self.network = network
        self.best_params = None

        # Index of the variable to check in the monitoring extension
        self.var_idx = monitor.output_links[var][idx]

        self.best_value = np.inf

    def execute(self, batch_id):
        current_value = self.val_monitor.history[-1][self.var_idx]
        if current_value < self.best_value:
            self.best_value = current_value
            self.waiting = 0
            self.absolute_waiting = 0
            if self.network:
                self.best_params = get_all_param_values(self.network)
        else:
            self.waiting += 1
            self.absolute_waiting += 1
        strs = ['Learning rate: {}, waiting {}/{}, absolute waiting {}/{}, '
                'best {}'.format(
            self.lr.get_value(), self.waiting, self.patience,
            self.absolute_waiting, self.absolute_patience, self.best_value)]
        if self.waiting > self.patience:
            self.lr.set_value(self.lr.get_value()/self.decay_rate)
            self.waiting = 0
            msg = 'Learning rate decreased'
            if self.network:
                set_all_param_values(self.network, self.best_params)
                msg += '... best network re-loaded'
            strs.append(msg)
        return strs

    def check_condition_virtual(self, batch_id):
        res = False
        if self.absolute_waiting > self.absolute_patience:
            res = 'Patience exceeded'
        elif self.lr.get_value() < self.min_value:
            res = 'Learning rate too small'

        if res and self.network:
            set_all_param_values(self.network, self.best_params)
            res += '... best network re-loaded'

        if res:
            res = [res]
        return res


class Saver(Extension):
    """Extension to pickle objects.

    Only the compute_object method should be overwritten.
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
        raise NotImplementedError


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

        self.var_names = []
        for var in self.var_monitor.monitored_var:
            if isinstance(var, MonitoredQuantity):
                self.var_names.extend(var.names)
            else:
                self.var_names.append(var.name)

    def compute_object(self):
        np_history = np.array(self.var_monitor.history)
        np_iterations = np.array(self.var_monitor.iterations)
        return ((np_iterations, np_history, self.var_names),
                ['Variable histories dumped into {}'.format(self.folder_path)])
