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
    __init__ and execute_virtual. Potentially, if you want to execute the
    extension if a condition is met, you can implement the condition method as
    well.

    Extension objects are provided to a :class:`Trainer` object.

    Parameters
    ----------
    name_extension: string
        The name of the extension
    freq: int or 'epoch' or None
        The frequency with which the extension is called. If None, it is never
        called (or maybe at the beginning or the end of training). If 'epoch',
        it is called at the end of each epoch.
    apply_at_the_start: bool, default False
        Apply the extension at the start of training
    apply_at_the_end: bool, default False
        Apply the extension at the end of training or when training is
        interrupted
    """
    def __init__(self, name_extension, freq,
                 apply_at_the_end=False, apply_at_the_start=False):
        self.name_extension = name_extension
        self.freq = freq
        self.apply_at_the_end = apply_at_the_end
        self.apply_at_the_start = apply_at_the_start
        self.total_spent_time_in_ext = 0

    def check(self, batch_id, end_epoch=False):
        """This method is called by the :class:`Trainer` object at every batch
        during training. It checks if the execution can be executed depending
        on its frequency and on its condition.

        Returns
        -------
        Boolean
            True if the execution can be existed. False otherwise.
        """
        if not self.freq:
            return False

        if self.freq == 'epoch':
            return end_epoch and self.condition(batch_id)

        return batch_id % self.freq and self.condition(batch_id)

    def condition(self, batch_id):
        """The extension might only be run if certain conditions are met.
        """
        return True

    def execute(self, batch_id):
        """The method that is called when the extension is executed.

        Do not re-implement this method but execute_virtual instead.

        Returns
        -------
        scalar:
            The time of the execution
        list of strings
            The lines to be printed during training. They will automatically be
            indented.
        """
        ts = time.time()
        result = self.execute_virtual(batch_id)
        te = time.time()
        self.total_spent_time_in_ext += te-ts
        return te-ts, result

    def execute_virtual(self, batch_id):
        """The method which should be re-implemented.
        """
        return ['Extension was executed']

    def start(self):
        return self.execute(0)

    def finish(self, batch_id):
        return self.execute(batch_id)


class EndCondition(object):
    """Abstract class responsible for terminating the training.

    EndCondition are registered to a Trainer object.

    check_condition_virtual is the only method to be re-implemented if you
    inherit from this class. Check MaxIteration for a simple exemple.

    Parameters:
    -----------
    name: string
        Name of the ending condition.
    freq: int or 'epoch'
        Frequency to which this ending condition should be checked.
    """
    def __init__(self, name, freq):
        self.name = name
        self.freq = freq

    def check_condition(self, batch_id, end_epoch=False):
        if (end_epoch and self.freq == 'epoch') or \
                (self.freq != 'epoch' and not batch_id % self.freq):
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
    def __init__(self, max_batchs):
        EndCondition.__init__(self, 'Max Iteration', 1)
        self.max_batchs = max_batchs

    def check_condition_virtual(self, batch_id):
        if batch_id > self.max_batchs:
            return ['Maximal number of batches reached']
        return False


class MaxTime(EndCondition):
    """Stops training when a certain amount of training time is reached
    """
    def __init__(self, max_time=3600*48):
        EndCondition.__init__(self, 'Max Iteration', 1)
        self.max_time = max_time
        self.begin_time = time.time()

    def check_condition_virtual(self, batch_id):
        if (time.time() - self.begin_time) > self.max_time:
            return ['Time exceeded']
        return False


class Monitor(Extension):
    """
    Base class for monitoring different metrics, which can be:
    - external metrics that do not depend on theano tensors: you should use
        :class:`ExternalMetricMonitor`.
    - metrics that depend directly on theano tensors computed during training:
        you should use :class:`TrainMonitor`.
    - metrics that depend on theano tensors computed on another dataset or a
        data generator: you should use :class:`ValidationMonitor`.
    """
    def __init__(self, name_extension, freq, metrics, **kwargs):
        Extension.__init__(self, name_extension, freq, **kwargs)
        self.metrics = metrics
        self.metric_names = [m.name for m in metrics]
        if None in self.metric_names:
            raise Exception('A metric provided does not have a name. Set it'
                            'with metric.name="zoulou"')

    def find_metric_from_name(self, metric_name):
        """
        Returns the monitored metric given its name
        """
        for metric in self.metrics:
            if metric.name == metric_name:
                return metric
        raise ValueError('No metric found for name {}'.format(metric_name))


class ExternalMetricMonitor(Monitor):
    """Extension to monitor MonitoredQuantity metrics that don't depend
    on theano tensors.

    Parameters:
    -----------
    monitored_quantities: list of MonitoredQuantity objets
        the list of special metrics to be monitored. These metrics should not
        require any theano tensor values to be computed.
    """
    def __init__(self, name_extension, freq, metrics, **kwargs):
        Monitor.__init__(self, name_extension, freq, metrics, **kwargs)

        # Will store the time required to compute each individual metric
        self.current_spent_time = np.zeros(len(self.metrics))

        # Stores all the values of the monitored metrics.
        self.history = []
        self.iterations = []

    def execute_virtual(self, batch_id):
        self.current_spent_time = np.zeros(len(self.metrics))

        # Compute the metrics from the tensor values
        metric_values = []
        for i, metric in enumerate(self.metrics):
            begin = time.time()
            res = metric.calculate()
            self.current_spent_time[i] = time.time() - begin
            if not isinstance(res, list):
                res = [res]
            metric_values.extend(res)

        metric_values = np.array(metric_values)

        strs = self.get_str(metric_values)
        self.history.append(metric_values)
        self.iterations.append(batch_id)
        return strs

    def get_str(self, metric_values):
        strs = []
        c = 0
        for timing, metric in zip(self.current_spent_time, self.metrics):
            strs.append('Computed in {:.3g} seconds:'.format(timing))
            metric.write_str(strs, metric_values[c:c+metric.n_outputs], '  ')
            c += metric.n_outputs

        return strs


class MetricMonitor(Monitor):
    """Extension to monitor metrics, which are either theano tensors or
    :class:`MonitoredQuantity` objects.

    It compiles a theano function internally.

    This is an abstract class.

    Parameters
    ----------
    inputs: list of theano tensors
        tensors necessary to compute the metrics
    metrics_with_aggfun : a list of either (a) theano tensors or
        MonitoredQuantity objects or (b) a tuples of (theano tensor or
        MonitoredQuantity object, aggregation function)
        the list of metrics that are monitored by this extension.
        Aggregation functions allow to modify the way the scores are
        aggregated over minibatches. If not provided, the default function is
        lambda x, n_batches: x / float(n_batches)
    output_tensors
    updates: list of theano updates, optional, default=None
        Updates fo be performed by the theano function.
    """
    def __init__(self, name_extension, freq, inputs, metrics_with_aggfun,
                 updates=None, givens=None, **kwargs):

        # Divide monitored metrics and corresponding aggregation schemes
        aggregation_functions = []
        metrics = []
        default_agg = lambda x, n_batches: x / float(n_batches)
        for i, metric_aggfun in enumerate(metrics_with_aggfun):
            if isinstance(metric_aggfun, tuple):
                aggregation_functions.append(metric_aggfun[1])
                metrics.append(metric_aggfun[0])
            else:
                aggregation_functions.append(default_agg)
                metrics.append(metric_aggfun)

        self.agg_fun = aggregation_functions

        Monitor.__init__(self, name_extension, freq, metrics,
                         **kwargs)

        # Total number of outputs (some monitored metrics may have several
        # outputs). output_links {metric: indices of outputs} links each metric
        # to the indices of its corresponding outputs
        self.n_outputs = 0
        self.output_links = {}
        for metric in metrics:
            if isinstance(metric, MonitoredQuantity):
                n = metric.n_outputs
            elif isinstance(metric, theano.Variable):
                n = 1
            self.output_links[metric] = range(self.n_outputs, self.n_outputs + n)
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

        # input_links is a dictionary {metric: list of indices}. Each element
        # contains the indices of the tensors from required_tensors required
        # to compute a metric from metrics
        self.input_links = {}
        for metric in metrics:
            links = []
            if isinstance(metric, MonitoredQuantity):
                for t in metric.required_tensors:
                    add_tensor(t, links)
            elif isinstance(metric, theano.Variable):
                add_tensor(metric, links)
            else:
                raise ValueError('metrics should contain either '
                                 'theano tensors or MonitoredQuantity objects')
            self.input_links[metric] = links

        # Function that will output the values of the required tensors for
        # given inputs.
        self.f = theano.function(inputs, self.required_tensors,
                                 updates=updates, givens=givens,
                                 on_unused_input='warn')

        # Stores all the values of the monitored metrics.
        self.history = []
        self.iterations = []

    def execute_virtual(self, batch_id):
        metric_values = self.compute_metrics()
        strs = self.get_str(metric_values)
        self.history.append(metric_values)
        self.iterations.append(batch_id)
        return strs

    def compute_metrics(self):
        """Computes and return the values of the metrics over the whole data
        generator. Called by the execute_virtual method.
        """
        raise NotImplementedError

    def compute_metrics_minibatch(self, *inputs):
        """Computes and return the values of the metrics for given inputs of
        a given minibatch.
        """
        # List of values of the required tensors. We have to compute the
        # metrics from them.
        tensor_values = self.f(*inputs)

        # Compute the metrics from the tensor values
        metric_values = []
        for metric, agg_scheme in zip(self.metrics, self.agg_fun):
            values = [tensor_values[i] for i in self.input_links[metric]]
            if isinstance(metric, theano.Variable):
                res = values[0]
            else:
                res = metric.calculate(*values)
            if not isinstance(res, (list, tuple)):
                res = [res]
            metric_values.extend(res)

        return np.array(metric_values)

    def get_str(self, metric_values):
        strs = []
        c = 0
        for metric in self.metrics:
            if isinstance(metric, MonitoredQuantity):
                metric.write_str(strs, metric_values[c:c+metric.n_outputs])
                c += metric.n_outputs
            else:
                strs.append(metric.name + ': {}'.format(metric_values[c]))
                c += 1

        return strs


class ValidationMonitor(MetricMonitor):
    """
    Extension to monitor metrics computed on a dataset. These metrics can
    either be tensors or MonitoredQuantity objects.
    """
    def __init__(self, name_extension, freq, inputs, metrics,
                 data_generator, updates=None, givens=None,
                 apply_at_the_end=True, apply_at_the_start=False,
                 init_states=None, **kwargs):
        MetricMonitor.__init__(
            self, name_extension, freq, inputs, metrics,
            updates=updates, givens=givens,
            apply_at_the_end=apply_at_the_end,
            apply_at_the_start=apply_at_the_start, **kwargs)
        self.data_generator = data_generator
        self.init_states = init_states

    def compute_metrics(self):

        # Save initial states to restore them later
        if self.init_states:
            init_values = []
            for init_state in self.init_states:
                init_values.append(init_state.get_value())
                init_state.set_value(.0*init_state.get_value())

        c = 0.0
        metric_values = np.zeros(self.n_outputs, dtype=floatX)
        for data in self.data_generator():
            metric_values += self.compute_metrics_minibatch(*data)
            c += 1

        for i, agg_fun in enumerate(self.agg_fun):
            metric_values[i] = agg_fun(metric_values[i], c)

        # Restore initial states
        if self.init_states:
            for init_state, init_value in zip(self.init_states, init_values):
                init_state.set_value(init_value)

        return metric_values


class TrainMonitor(MetricMonitor):
    """
    Extension required by `class:Trainer` to process_batch updates and monitor
    metrics (either tensors or MonitoredQuantity objects).
    """
    def __init__(self, freq, inputs, metrics, updates, givens=None,
                 **kwargs):
        MetricMonitor.__init__(self, 'Training', freq, inputs, metrics,
                               updates, givens=givens, **kwargs)

        self.time_since_last_execute = 0

        # Needs to keep track of the computed values during training
        self.current_metric_values = np.zeros(self.n_outputs, dtype=floatX)
        # Needs also to keep track of the number of minibatches since last
        # display
        self.n_minibatches = 0

    def execute(self, batch_id):

        begin = time.time()
        logs = self.execute_virtual(batch_id)
        self.time_since_last_execute += (time.time() - begin)

        timing = self.time_since_last_execute
        self.total_spent_time_in_ext += timing
        self.time_since_last_execute = 0

        return timing, logs

    def compute_metrics(self):

        for i, agg_fun in enumerate(self.agg_fun):
            self.current_metric_values[i] = agg_fun(
                self.current_metric_values[i], self.n_minibatches)

        # Reset current metric values for next pass
        res = self.current_metric_values
        self.current_metric_values = np.zeros(self.n_outputs, dtype=floatX)
        self.n_minibatches = 0

        return res

    def train(self, *inputs):
        begin = time.time()
        self.current_metric_values += self.compute_metrics_minibatch(*inputs)
        self.n_minibatches += 1
        self.time_since_last_execute += (time.time() - begin)


class LearningRateDecay(Extension, EndCondition):
    """
    Both extension and ending condition that decreases the learning rate if
    there is no improvement on a metric monitored by a monitoring extension.
    If does not improve for absolute_patience, then the training stops.

    The frequence of this extension is the same as the :class:`Monitor` monitor
    parameter.

    Parameters:
    -----------
    monitor: :class:`Monitor` object
        :class:`Monitor` object which computes the metrics you are
        interested in.
    metric: MonitoredQuantity or tensor
        the metric (either MonitoredQuantity object or tensor) that
        you are interested in.
    idx: int, default=0
        if metric computes several outputs, this index selects a single one.
    learning_rate: theano shared variable
        the variable storing the current learning rate
    patience: int, default=5
        the number of times we allow the metric to not improve before
        decreasing the learning rate
    max_patience: int, default=7
        the number of times we allow the metric to not improve before we
        stop the training.
    decay_rate: float, default=2.0
        the rate at which the learning rate is decreased
    min_value: float
        the minimal value that we tolerate for the learning rate. Below it, we
        stop training.
    params: list of shared variables (default None)
        if you want the best parameters to be saved and restored. If you want
        to include both the parameters of the network and those of the
        optimisation algorithm (such as momentum), you may want to give
        params=list(updates.keys()) as input.
    """
    def __init__(self, monitor, metric_name, learning_rate, idx=0, patience=5,
                 max_patience=7, decay_rate=2., min_value=1e-12, params=None,
                 metric_mode='min'):
        Extension.__init__(self, 'Learning rate', monitor.freq)
        EndCondition.__init__(self, 'Learning rate', monitor.freq)
        self.lr = learning_rate
        self.patience = patience
        self.absolute_patience = max_patience
        self.decay_rate = decay_rate
        self.waiting = 0
        self.absolute_waiting = 0
        self.validation_monitor = monitor
        self.min_value = min_value

        self.params = params
        self.best_params = [p.get_value() for p in self.params]

        self.metric = monitor.find_metric_from_name(metric_name)
        # Index of the metric to check in the monitoring extension
        self.metric_idx = monitor.output_links[self.metric][idx]
        self.mode_metric = metric_mode
        if metric_mode == 'max':
            self.m = -1
        elif metric_mode == 'min':
            self.m = 1
        else:
            raise ValueError

        self.best_value = self.m * np.inf

    def execute_virtual(self, batch_id):
        current_value = self.validation_monitor.history[-1][self.metric_idx]
        if np.isnan(current_value):
            raise Exception('nan detected')

        if current_value * self.m < self.best_value * self.m:
            self.best_value = current_value
            self.waiting = 0
            self.absolute_waiting = 0
            if self.params:
                self.best_params = [p.get_value() for p in self.params]
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
            if self.params:
                for p, v in zip(self.params, self.best_params):
                    p.set_value(v)
                msg += '... best network re-loaded'
            strs.append(msg)
        return strs

    def check_condition_virtual(self, batch_id):
        res = False
        if self.absolute_waiting > self.absolute_patience:
            res = 'Patience exceeded'
        elif self.lr.get_value() < self.min_value:
            res = 'Learning rate too small'

        if res and self.params:
            for p, v in zip(self.params, self.best_params):
                p.set_value(v)
            res += '... best network re-loaded'

        if res:
            res = [res]
        return res


class LearningRateDecay2(Extension, EndCondition):
    def __init__(self, init_lr, end_lr, freq, n_batches):
        Extension.__init__(self, 'Learning rate', freq)
        EndCondition.__init__(self, 'Learning rate', freq)
        self.n_batches = n_batches
        self.lr = init_lr
        self.decay_rate = np.float32((end_lr / init_lr.get_value()) ** (
            float(freq) / n_batches))

    def execute_virtual(self, batch_id):
        self.lr.set_value(self.lr.get_value() * self.decay_rate)
        strs = ['New learning rate: {}'.format(self.lr.get_value())]
        return strs

    def check_condition_virtual(self, batch_id):
        res = False
        if batch_id > self.n_batches:
            res = ['Learning rate too small']
        return res


class Saver(Extension):
    """Extension to pickle objects.

    Only the compute_object method should be overwritten.
    """
    def __init__(self, name_extension, freq, folder_path, file_name,
                 apply_at_the_end=True, **kwargs):
        super(Saver, self).__init__(name_extension, freq,
                                    apply_at_the_end=apply_at_the_end, **kwargs)
        self.folder_path = folder_path
        self.file_name = file_name

    def execute_virtual(self, batch_id):
        return self.save()

    def save(self, file_path=None):
        if not file_path:
            file_path = os.path.join(self.folder_path, self.file_name + '.pkl')

        file_handle = open(file_path, 'wb')
        obj, msg = self.compute_object()
        cPickle.dump(obj, file_handle)
        return msg

    def compute_object(self):
        """
        It should return a tuple of two elements. The first is the object to be
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
        return (get_all_param_values(self.net),
                ['Network dumped into {}'.format(self.folder_path)])


class BestNetworkSaver(Saver):
    """Saves the parameters of the network.
    """
    def __init__(self, params, monitor, metric_name, folder_path,
                 restore_best_weights_at_the_end=True, idx=0,
                 file_name='best_net', apply_at_the_end=True,
                 metric_mode='min', dont_save_for_first_n_it=None):
        super(BestNetworkSaver, self).__init__('Best Network Saver',
                                               monitor.freq, folder_path,
                                               file_name, apply_at_the_end)
        self.params = params
        self.best_params_values = [p.get_value() for p in params]

        self.validation_monitor = monitor

        metric = monitor.find_metric_from_name(metric_name)
        # Index of the metric to check in the monitoring extension
        self.metric_idx = monitor.output_links[metric][idx]
        self.mode_metric = metric_mode
        if metric_mode == 'max':
            self.m = -1
        elif metric_mode == 'min':
            self.m = 1
        else:
            raise ValueError

        self.restore_best_weights_at_the_end = restore_best_weights_at_the_end

        self.best_value = self.m * np.inf

        self.dont_save_for_first_n_it = dont_save_for_first_n_it
        self.n_times_checked = 0

    def condition(self, batch_id):
        # Check if dont_save_for_first_n_it has passed
        self.n_times_checked += 1
        if not self.dont_save_for_first_n_it or (
                    self.n_times_checked < self.dont_save_for_first_n_it):
            return False

        # Check if the validation monitor has indeed recorded values
        if not self.validation_monitor.history:
            raise Exception('Best network saver should be placed after the'
                            'validation monitor in the list of extensions'
                            'provided to the Trainer object.')

        current_value = self.validation_monitor.history[-1][self.metric_idx]
        if self.m * current_value < self.m * self.best_value:
            self.best_value = current_value
            self.best_params_values = [p.get_value() for p in self.params]
            return True
        return False

    def compute_object(self):
        return (self.best_params_values,
                ['Best Network dumped into {}'.format(self.folder_path)])

    def restore(self, file_path=None):
        if not file_path:
            file_path = os.path.join(self.folder_path, self.file_name + '.pkl')
        file_handle = open(file_path, 'r')
        self.best_params_values = cPickle.load(file_handle)

        for p, v in zip(self.params, self.best_params_values):
            p.set_value(v)

    def finish(self, batch_id):
        self.dont_save_for_first_n_it = 0
        self.execute(batch_id)
        if not self.restore_best_weights_at_the_end:
            return None, None

        b = time.time()
        for p, v in zip(self.params, self.best_params_values):
            p.set_value(v)
        e = time.time()
        return e-b, ['... best network re-loaded']


class MetricSaver(Saver):
    """Saves the history of a ValidationMonitor extension
    """
    def __init__(self, metric_monitor, freq, folder_path, name=None):
        if not name:
            name = metric_monitor.name_extension
        super(MetricSaver, self).__init__('Metric saver ' + name, freq,
                                          folder_path, 'metric_saver_' + name)
        self.metric_monitor = metric_monitor

        self.metric_names = []
        for metric in self.metric_monitor.metrics:
            if isinstance(metric, MonitoredQuantity):
                self.metric_names.extend(metric.names)
            else:
                self.metric_names.append(metric.name)

    def compute_object(self):
        np_history = np.array(self.metric_monitor.history)
        np_iterations = np.array(self.metric_monitor.iterations)
        d = {'iterations': np_iterations,
             'history': np_history,
             'names': self.metric_names}
        return (d,
                [self.name_extension +
                 ' Metric histories dumped into {}'.format(
                     self.folder_path)])
