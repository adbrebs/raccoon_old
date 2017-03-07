import os
import time
import cPickle
import numpy as np
import theano
import theano.tensor as T
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

    def check(self, batch_id, epoch_id, end_epoch=False):
        """This method is called by the :class:`Trainer` object at every batch
        during training. It checks if the execution can be executed depending
        on its frequency and on its condition.

        Returns
        -------
        Boolean
            True if the execution can be existed. False otherwise.
        """
        if not self.freq:
            freq_cond = False

        elif self.freq == 'epoch':
            freq_cond = end_epoch  # and self.condition(batch_id, epoch_id)

        else:
            freq_cond = not (batch_id % self.freq)

        return self.condition(freq_cond, batch_id, epoch_id)

    def condition(self, freq_cond, batch_id, epoch_id):
        """The extension might only be run if certain conditions are met.

        freq_cond: boolean
            True if the frequency condition is satisfied. False otherwise.
        """
        ts = time.time()
        decision = self.condition_virtual(freq_cond, batch_id, epoch_id)
        te = time.time()
        self.total_spent_time_in_ext += te - ts
        return decision

    def condition_virtual(self, freq_cond, batch_id, epoch_id):
        """The extension might only be run if certain conditions are met.
        """
        return freq_cond

    def execute(self, batch_id, epoch_id=None):
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
        msg = self.execute_virtual(batch_id, epoch_id)
        te = time.time()
        self.total_spent_time_in_ext += te - ts
        return te - ts, msg

    def execute_virtual(self, batch_id, epoch_id=None):
        """The method which should be re-implemented.
        """
        return ['Extension was executed']

    def start(self):
        return self.execute(0, 0)

    def finish(self, batch_id, epoch_id):
        return self.execute(batch_id, epoch_id)


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

    def check_condition(self, batch_id, epoch_id, end_epoch=False):
        if (end_epoch and self.freq == 'epoch') or \
                (self.freq != 'epoch' and not batch_id % self.freq):
            return self.check_condition_virtual(batch_id, epoch_id)
        return False

    def check_condition_virtual(self, batch_id, epoch_id):
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

    def __init__(self, max_batchs=np.inf, max_epochs=np.inf):
        EndCondition.__init__(self, 'Max Iteration', 1)
        self.max_batchs = max_batchs
        self.max_epochs = max_epochs
        if not max_batchs and not max_epochs:
            raise Exception('Either max_batchs or max_epochs should be set.')

    def check_condition_virtual(self, batch_id, epoch_id):
        if batch_id > self.max_batchs:
            return ['Maximal number of batches reached']
        if epoch_id > self.max_epochs:
            return ['Maximal number of epochs reached']
        return False


class MaxTime(EndCondition):
    """Stops training when a certain amount of training time is reached
    """

    def __init__(self, max_time=3600 * 48):
        EndCondition.__init__(self, 'Max Iteration', 1)
        self.max_time = max_time
        self.begin_time = time.time()

    def check_condition_virtual(self, batch_id, epoch_id):
        if (time.time() - self.begin_time) > self.max_time:
            return ['Time exceeded']
        return False


class NoTraining(EndCondition):
    """Stops training when a maximal number of iterations is reached.
    """

    def __init__(self, monitor, metric_name, idx=0, patience=5):
        EndCondition.__init__(self, 'No training', monitor.freq)

        self.monitor = monitor
        self.patience = patience
        self.metric_name = metric_name
        self.metric = monitor.find_metric_from_name(metric_name)
        # Index of the metric to check in the monitoring extension
        self.metric_idx = monitor.output_links[self.metric][idx]

        self.counter = 0
        self.initial_value = None

    def check_condition_virtual(self, batch_id, epoch_id):

        # Check if the validation monitor has indeed recorded values
        if not self.monitor.history:
            raise Exception('The no training extension should be placed after '
                            'the validation monitor in the list of extensions'
                            'provided to the Trainer object.')

        current_value = self.monitor.history[-1][self.metric_idx]

        if not self.initial_value:
            self.initial_value = current_value
            return False

        if self.initial_value == current_value:
            self.counter += 1

        if self.counter > self.patience:
            return ['Training has not started.']

        return False


# class ExternalMetricMonitor(Monitor):
#     """Extension to monitor MonitoredQuantity metrics that don't depend
#     on theano tensors.
#
#     Parameters:
#     -----------
#     monitored_quantities: list of MonitoredQuantity objets
#         the list of special metrics to be monitored. These metrics should not
#         require any theano tensor values to be computed.
#     """
#
#     def __init__(self, name_extension, freq, metrics, **kwargs):
#         Monitor.__init__(self, name_extension, freq, metrics, **kwargs)
#
#         # Will store the time required to compute each individual metric
#         self.current_spent_time = np.zeros(len(self.metrics))
#
#         # Stores all the values of the monitored metrics.
#         self.history = []
#         self.iterations = []
#
#     def execute_virtual(self, batch_id, epoch_id=None):
#         self.current_spent_time = np.zeros(len(self.metrics))
#
#         # Compute the metrics from the tensor values
#         metric_values = []
#         for i, metric in enumerate(self.metrics):
#             begin = time.time()
#             res = metric.calculate()
#             self.current_spent_time[i] = time.time() - begin
#             if not isinstance(res, list):
#                 res = [res]
#             metric_values.extend(res)
#
#         metric_values = np.array(metric_values)
#
#         strs = self.get_str(metric_values)
#         self.history.append(metric_values)
#         self.iterations.append(batch_id, epoch_id)
#         return strs
#
#     def get_str(self, metric_values):
#         strs = []
#         c = 0
#         for timing, metric in zip(self.current_spent_time, self.metrics):
#             strs.append('Computed in {:.3g} seconds:'.format(timing))
#             metric.write_str(strs, metric_values[c:c + metric.n_outputs], '  ')
#             c += metric.n_outputs
#
#         return strs


class MetricMonitor(Extension):
    """Extension to monitor metrics, which are either theano tensors or
    :class:`MonitoredQuantity` objects.

    It compiles a theano function internally.

    This is an abstract class with two children classes:
    - :class:`TrainMonitor`: the extension responsible for training and
        monitoring metrics on the training dataset
    - :class:`ValidationMonitor`: an extension for monitoring metrics on a
        given validation dataset dataset or a data generator.

    Parameters
    ----------
    inputs: list of theano tensors
        tensors necessary to compute the metrics
    metric_list: a list of either (a) theano tensors or MonitoredQuantity
            objects or (b) dict {'metric': tensor or MonitoredQuantity,
            'counter': scalar tensor (optional), 'agg_fun': elemwise function
            of two arguments (optional), 'norm_fun': elemwise function of
            two arguments (optional)}
        the list of metrics that are monitored by this extension.
        If a dict is provided, the key 'metric' has to be provided and
        represents the metric. The other keys are optional:
        - 'counter': the counter, a quantity by which the
            aggregated metric divided, before 'norm_fun' is applied. If not
            provided, its default value is default_counter.
        - 'agg_fun': a function that takes two minibatch metric values and
            aggregate them. default: add
        - 'norm_fun': a function that transforms the total metric and the
            total counter. default: lambda a, b: a / b
        The precise processing pattern is sketched below:

            total_metric = 0
            total_counter = 0
            for minibatch in generator:
                metric, counter = compute_metric(minibatch)
                total_metric = agg_fun(total_metric, metric)
                total_counter += counter

            final_metric_value = norm_fun(total_metric, total_counter)

    default_counter: int or tensor variable (default 1)
        If a provided metric is not a dictionary or does not have a 'counter'
        key, it will use default_counter as counter.

    inputs: list of theano tensors
        the tensor inputs required to compute the metrics
    updates: list of theano updates, optional, default=None
        Updates fo be performed by the theano function.
    custom_process_fun: a function taking as input the data generator output
        and returning a processed version of it. It is typically used to reset
        initial states of shared variables. This function is called before
        processing a minibatch. For example, it could be:
        def custom_process_fun(generator_output):
            inputs, new_seq = generator_output
            if new_seq:
                model.reset_shared_init_states([h_ini, k_ini, w_ini])
            return inputs
    """

    def __init__(self, name_extension, freq, inputs, metric_list,
                 default_counter=1, updates=None, givens=None,
                 custom_process_fun=None, **kwargs):
        Extension.__init__(self, name_extension, freq, **kwargs)

        # Divide monitored metrics and corresponding aggregation schemes
        self.metrics = metrics = []
        self.metric_names = []
        counters = []
        self.agg_funs = []
        self.norm_funs = []
        self.updates = updates
        self.custom_process_fun = custom_process_fun

        for i, metric_dict in enumerate(metric_list):
            if not isinstance(metric_dict, dict):
                metric_dict = {'metric': metric_dict}

            if 'counter' not in metric_dict:
                if not default_counter:
                    raise Exception('A default batch size should be provided.')
                metric_dict['counter'] = default_counter

            if isinstance(metric_dict['counter'], int):
                metric_dict['counter'] = (np.array(metric_dict['counter']) *
                                          T.ones((1,), dtype=floatX))[0]

            metric = metric_dict['metric']
            metrics.append(metric)
            if isinstance(metric, MonitoredQuantity):
                self.metric_names.extend(metric.names)
            else:
                self.metric_names.append(metric.name)
            counters.append(metric_dict['counter'])
            self.agg_funs.append(metric_dict.get('agg_fun', np.add))
            self.norm_funs.append(metric_dict.get('norm_fun', lambda a, b: a / b))

        if None in self.metric_names:
            raise Exception('A metric provided does not have a name. Set it'
                            'with metric.name="zoulou"')

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
            self.output_links[metric] = range(self.n_outputs,
                                              self.n_outputs + n)
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
                a_list.append(len(self.required_tensors) - 1)

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

        # Adds the counters to required function
        self.required_tensors.extend(counters)

        # Function that will output the values of the required tensors for
        # given inputs.
        self.f = theano.function(inputs, self.required_tensors,
                                 updates=updates, givens=givens,
                                 on_unused_input='warn')

        # Stores all the values of the monitored metrics.
        self.history = []
        self.iterations = []

    def find_metric_from_name(self, metric_name, return_mode='raise'):
        """
        Returns the monitored metric given its name

        return_mode: string {'raise', 'None'}
        """
        for metric in self.metrics:
            if isinstance(metric, MonitoredQuantity):
                if metric_name in metric.names:
                    return metric
            else:
                if metric.name == metric_name:
                    return metric

        if return_mode == 'raise':
            raise ValueError('No metric found for name {}'.format(metric_name))
        elif return_mode == 'None':
            return None
        else:
            raise ValueError

    def execute_virtual(self, batch_id, epoch_id=None):
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

        if self.custom_process_fun:
            inputs = self.custom_process_fun(inputs)

        # List of values of the required tensors. We have to compute the
        # metrics from them.
        tensor_values = self.f(*inputs)
        counter_values = np.array(tensor_values[-len(self.metrics):])
        tensor_values = tensor_values[:-len(self.metrics)]

        # Compute the metrics from the tensor values
        metric_values = []
        for metric in self.metrics:
            values = [tensor_values[i] for i in self.input_links[metric]]
            if isinstance(metric, theano.Variable):
                res = values[0]
            else:
                res = metric.calculate(*values)
            if not isinstance(res, (list, tuple)):
                res = [res]
            metric_values.extend(res)

        return np.array(metric_values), counter_values

    def get_str(self, metric_values):
        strs = []
        c = 0
        for metric in self.metrics:
            if isinstance(metric, MonitoredQuantity):
                metric.write_str(strs, metric_values[c:c + metric.n_outputs])
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

    def __init__(self, name_extension, freq, inputs, metric_list,
                 data_generator, default_counter=1,
                 updates=None, givens=None, apply_at_the_end=True,
                 apply_at_the_start=False, init_states=None,
                 custom_process_fun=None, **kwargs):
        MetricMonitor.__init__(
            self, name_extension, freq, inputs, metric_list,
            default_counter=default_counter, updates=updates,
            givens=givens, apply_at_the_end=apply_at_the_end,
            apply_at_the_start=apply_at_the_start,
            custom_process_fun=custom_process_fun, **kwargs)
        self.data_generator = data_generator
        self.init_states = init_states

    def compute_metrics(self):

        # Save initial states to restore them later
        if self.init_states:
            init_values = []
            for init_state in self.init_states:
                init_values.append(init_state.get_value())
                init_state.set_value(.0 * init_state.get_value())

        metric_values = np.zeros(self.n_outputs, dtype=floatX)
        counter_values = np.zeros(self.n_outputs, dtype=floatX)

        for data in self.data_generator():
            m_values, c_values = self.compute_metrics_minibatch(*data)
            for i, agg_fun in enumerate(self.agg_funs):
                metric_values[i] = agg_fun(metric_values[i], m_values[i])
            counter_values += c_values

        for i, norm_fun in enumerate(self.norm_funs):
            metric_values[i] = norm_fun(metric_values[i], counter_values[i])

        # Restore initial states
        if self.init_states:
            for init_state, init_value in zip(self.init_states, init_values):
                init_state.set_value(init_value)

        return metric_values


ValidMonitor = ValidationMonitor


class TrainMonitor(MetricMonitor):
    """
    Extension required by `class:Trainer` to process_batch updates and monitor
    metrics (either tensors or MonitoredQuantity objects).

    train_freq: the frequency at which the train method is called. freq=1 for
        training at every minibatch.
    """

    def __init__(self, freq, inputs, metric_list, updates,
                 default_counter=1, givens=None, train_freq=1,
                 custom_process_fun=None, **kwargs):
        MetricMonitor.__init__(
            self, 'Training', freq, inputs, metric_list,
            default_counter=default_counter,
            updates=updates, givens=givens,
            custom_process_fun=custom_process_fun, **kwargs)

        self.time_since_last_execute = 0

        # Needs to keep track of the computed values during training
        self.current_metric_values = np.zeros(self.n_outputs, dtype=floatX)
        self.current_metric_counters = np.zeros(self.n_outputs, dtype=floatX)
        # Needs also to keep track of the number of minibatches since last
        # display
        self.n_minibatches = 0
        self.train_freq = train_freq

    def execute(self, batch_id, epoch_id=None):
        begin = time.time()
        logs = self.execute_virtual(batch_id, epoch_id)
        self.time_since_last_execute += (time.time() - begin)

        timing = self.time_since_last_execute
        self.total_spent_time_in_ext += timing
        self.time_since_last_execute = 0

        return timing, logs

    def compute_metrics(self):

        res = np.zeros_like(self.current_metric_values)
        for i, norm_fun in enumerate(self.norm_funs):
            res[i] = norm_fun(self.current_metric_values[i],
                              self.current_metric_counters[i])

        # Reset current metric values for next pass
        self.current_metric_values = np.zeros(self.n_outputs, dtype=floatX)
        self.current_metric_counters = np.zeros(self.n_outputs, dtype=floatX)
        self.n_minibatches = 0

        return res

    def train(self, minibatch, *inputs):
        if minibatch % self.train_freq:
            return

        begin = time.time()

        m_values, c_values = self.compute_metrics_minibatch(*inputs)

        for i, agg_fun in enumerate(self.agg_funs):
            self.current_metric_values[i] = agg_fun(
                self.current_metric_values[i], m_values[i])
        self.current_metric_counters += c_values
        self.n_minibatches += 1
        self.time_since_last_execute += (time.time() - begin)


class ValidationSchedule(Extension, EndCondition):
    """
    Both extension and ending condition that performs an action if there is no
    improvement on a metric monitored by a monitoring extension.
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
    metric_mode: string, either 'min' or 'max', default='min'
        indicates if the metric should be minimized of maximized
    idx: int, default=0
        if metric computes several outputs, this index selects a single one.
    process_function: function with no arguments
        the function to be called if the metric is not improved
    patience: int, default=5
        the number of times we allow the metric to not improve before
        calling the process_function
    max_patience: int, default=7
        the number of times we allow the metric to not improve before we
        stop the training.
    params: list of shared variables (default None)
        if you want the best parameters to be saved and restored. If you want
        to include both the parameters of the network and those of the
        optimisation algorithm (such as momentum), you may want to give
        params=list(updates.keys()) as input.
    nan_monitor: monitor or None (default None)
        If monitor and nan are encountered in its history, reload the last
        parameters and call the process_function. If None, nan are ignored.
    minimum_improvement: float (default 1.0)
        Controls by how much the metric has to improve to reset the patience.
    name: string (default None)
        Name of the extension
    """

    def __init__(self, monitor, metric_name, process_function, idx=0, patience=5,
                 max_patience=7, params=None, metric_mode='min',
                 nan_monitor=None, minimum_improvement=.0,
                 name='Validation schedule'):
        Extension.__init__(self, name, monitor.freq)
        EndCondition.__init__(self, name, monitor.freq)
        self.process_function = process_function
        self.patience = patience
        self.absolute_patience = max_patience
        self.waiting = 0
        self.absolute_waiting = 0
        self.validation_monitor = monitor
        self.minimum_improvement = minimum_improvement

        self.params = params
        if params:
            self.best_params = [p.get_value() for p in self.params]

        self.metric_name = metric_name
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

        self.nan_monitor = nan_monitor
        self.best_value = self.m * np.inf

    def condition_virtual(self, freq_cond, batch_id, epoch_id):
        if not self.nan_monitor or not self.nan_monitor.history:
            return freq_cond

        if np.any(np.isnan(self.nan_monitor.history[-1])):
            return True

        return freq_cond

    def execute_virtual(self, batch_id, epoch_id=None):

        strs = []

        if (self.nan_monitor and self.nan_monitor.history and
                np.any(np.isnan(self.nan_monitor.history[-1]))):
            self.waiting = np.inf
            self.absolute_waiting = 0
            self.nan_monitor.history.pop()

        else:
            current_value = self.validation_monitor.history[-1][
                self.metric_idx]

            if (self.m * (self.best_value - current_value) >
                    self.minimum_improvement):
                self.waiting = 0
                self.absolute_waiting = 0
            else:
                self.waiting += 1
                self.absolute_waiting += 1

            if current_value * self.m < self.best_value * self.m:
                self.best_value = current_value
                if self.params:
                    self.best_params = [p.get_value() for p in self.params]
                    strs.append('Best params saved.')

        if self.waiting > self.patience:
            self.process_function()
            self.waiting = 0
            msg = 'Learning rate decreased'
            if self.params:
                for p, v in zip(self.params, self.best_params):
                    p.set_value(v)
                msg += '... best network re-loaded'
            strs.append(msg)

        strs.append(
            '{} waiting {}/{}, absolute waiting {}/{}, best {} = {}'.format(
                self.display_info(), self.waiting,
                self.patience, self.absolute_waiting, self.absolute_patience,
                self.metric_name, self.best_value))

        return strs

    def display_info(self):
        return ''

    def check_condition_virtual(self, batch_id, epoch_id):
        res = False
        if self.absolute_waiting > self.absolute_patience:
            res = 'Patience exceeded'

        if res and self.params:
            for p, v in zip(self.params, self.best_params):
                p.set_value(v)
            res += '... best network re-loaded'

        if res:
            res = [res]
        return res


class SharedVariableValidationSchedule(ValidationSchedule):
    """
    Both extension and ending condition that modifies shared variables if
    there is no improvement on a metric monitored by a monitoring extension.
    If does not improve for absolute_patience, then the training stops.

    The frequence of this extension is the same as the :class:`Monitor` monitor
    parameter.

    Check the docstring of mother class ValidationSchedule
    for more information.

    Parameters:
    -----------
    shared_variables: theano shared variable or list of shared variables
        the variable(s) to be modified
    decay_rate: float, default=2.0
        the rate at which the learning rate is decreased. (the variables are
        divided by this decay_rate.)
    min_value: float (default None)
        the minimal value that we tolerate for the shared variables.
        Below it, we stop training.
    max_value: float (default None)
        the maximal value that we tolerate for the shared variables.
        Above it, we stop training.
    See mother class ValidationSchedule for the description of the other
    parameters.
    """
    def __init__(self, monitor, metric_name, shared_variables, idx=0, patience=5,
                 max_patience=7, decay_rate=2., max_value=None,
                 min_value=None, params=None, metric_mode='min',
                 nan_monitor=None, minimum_improvement=.0,
                 name='Shared variables decay'):

        if not isinstance(shared_variables, (list, tuple)):
            shared_variables = [shared_variables]
        self.shared_variables = shared_variables
        self.decay_rate = decay_rate
        self.min_value = min_value
        self.max_value = max_value

        def process_function():
            for sh in self.shared_variables:
                sh.set_value(np.float32(sh.get_value() / self.decay_rate))

        ValidationSchedule.__init__(
            self, monitor, metric_name, process_function, idx=idx,
            patience=patience, max_patience=max_patience, params=params,
            metric_mode=metric_mode, nan_monitor=nan_monitor,
            minimum_improvement=minimum_improvement,
            name=name)

    def check_condition_virtual(self, batch_id, epoch_id):
        res = ValidationSchedule.check_condition_virtual(
            self, batch_id, epoch_id)
        if res:
            return res

        for sh in self.shared_variables:
            if self.min_value and sh.get_value() < self.min_value:
                return ['too small']
            elif self.max_value and sh.get_value() > self.max_value:
                return ['too big']

        return False

    def display_info(self):
        return '{},'.format([float(sh.get_value())
                             for sh in self.shared_variables])


class LearningRateDecayValidation(SharedVariableValidationSchedule):
    """
    Both extension and ending condition that decreases the learning rate if
    there is no improvement on a metric monitored by a monitoring extension.
    If does not improve for absolute_patience, then the training stops.

    The frequence of this extension is the same as the :class:`Monitor` monitor
    parameter.

    Check the docstring of mother class ValidationSchedule
    for more information.

    Parameters:
    -----------
    learning_rate: theano shared variable or list of shared variables
        the variable(s) storing the current learning rate(s)
    See mother class ValidationSchedule for the description of the other
    parameters.
    """
    def __init__(self, monitor, metric_name, learning_rate, idx=0, patience=5,
                 max_patience=7, decay_rate=2., min_value=1e-12, params=None,
                 metric_mode='min', nan_monitor=None, minimum_improvement=.0):

        SharedVariableValidationSchedule.__init__(
            self, monitor, metric_name, learning_rate, idx=idx,
            patience=patience, max_patience=max_patience,
            decay_rate=decay_rate, min_value=min_value, params=params,
            metric_mode=metric_mode, nan_monitor=nan_monitor,
            minimum_improvement=minimum_improvement,
            name='Learning rate decay')


class VariableSchedule(Extension):
    """
    Modify a variable at specific iterations specified by the user.
    The variable can either be a scalar shared variable or a list with a single
    scalar element.
    """
    def __init__(self, variable_name, var, iteration_ids, values):
        extension_name = variable_name + ' schedule'
        Extension.__init__(self, extension_name, 1)
        self.variable_name = variable_name
        self.var = var
        self.iteration_ids = iteration_ids
        self.values = values

        if isinstance(var, T.sharedvar.SharedVariable):
            def fun_setter(x, var):
                var.set_value(x)
            def fun_getter(var):
                var.get_value()
        elif isinstance(var, (tuple, list)):
            def fun_setter(x, var):
                var[0] = x
            def fun_getter(var):
                return var[0]
        else:
            raise ValueError

        self.fun_setter = fun_setter
        self.fun_getter = fun_getter

    def condition_virtual(self, freq_cond, batch_id, epoch_id):
        """The extension might only be run if certain conditions are met.
        """
        if batch_id in self.iteration_ids:
            return True
        return False

    def execute_virtual(self, batch_id, epoch_id=None):
        value = self.values[self.iteration_ids.index(batch_id)]
        self.fun_setter(value, self.var)
        return ['New {}: {}'.format(self.variable_name,
                                    self.fun_getter(self.var))]


class LearningRateSchedule(VariableSchedule):
    def __init__(self, shared_lr, iteration_ids, values_lr):
        VariableSchedule.__init__(
            self, 'Learning rate', shared_lr, iteration_ids, values_lr)


class LearningRateLinearRange(Extension, EndCondition):
    """
    Decay the learning from an initial value to an end value in `n_batches`.
    """
    def __init__(self, init_lr, end_lr, freq, n_batches):
        Extension.__init__(self, 'Learning rate linear range', freq)
        EndCondition.__init__(self, 'Learning rate linear range', freq)
        self.n_batches = n_batches
        self.lr = init_lr
        self.decay_rate = np.float32((end_lr / init_lr.get_value()) ** (
            float(freq) / n_batches))

    def execute_virtual(self, batch_id, epoch_id=None):
        self.lr.set_value(self.lr.get_value() * self.decay_rate)
        strs = ['New learning rate: {}'.format(self.lr.get_value())]
        return strs

    def check_condition_virtual(self, batch_id, epoch_id):
        res = False
        if batch_id > self.n_batches:
            res = ['Learning rate too small']
        return res


class LearningRateDecay(Extension):
    """
    Decay the learning by `decay` after every `freq` iterations
    """
    def __init__(self, init_lr, decay, decay_start_after, freq):
        Extension.__init__(self, 'Learning rate simple decay', freq)
        self.lr = init_lr
        self.decay_rate = decay
        self.decay_start_after = decay_start_after
        self.n_times_decayed = 0

    def condition_virtual(self, freq_cond, batch_id, epoch_id):
        """The extension might only be run if certain conditions are met.
        """
        if not freq_cond:
            return False

        self.n_times_decayed += 1
        return self.n_times_decayed > self.decay_start_after

    def execute_virtual(self, batch_id, epoch_id=None):
        self.lr.set_value(self.lr.get_value() * self.decay_rate)
        return ['New learning rate: {}'.format(self.lr.get_value())]


class Saver(Extension):
    """Extension to pickle objects.

    Only the compute_object method should be overwritten.
    """

    def __init__(self, name_extension, freq, folder_path, file_name,
                 apply_at_the_end=True, **kwargs):
        super(Saver, self).__init__(
            name_extension, freq, apply_at_the_end=apply_at_the_end, **kwargs)
        self.folder_path = folder_path
        if not os.path.isdir(folder_path):
            os.makedirs(folder_path)
        self.file_name = file_name

    def execute_virtual(self, batch_id, epoch_id=None):
        return self.save()

    def save(self, file_path=None):
        if not file_path:
            file_path = os.path.join(self.folder_path, self.file_name + '.pkl')

        file_handle = open(file_path, 'wb')
        obj, msg = self.compute_object()
        cPickle.dump(obj, file_handle)
        file_handle.close()
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
                 metric_mode='min', dont_save_for_first_n_it=None,
                 save_on_disk=True, freq=False):
        if not freq:
            freq = monitor.freq
        super(BestNetworkSaver, self).__init__('Best Network Saver',
                                               freq, folder_path,
                                               file_name, apply_at_the_end)

        self.save_on_disk = save_on_disk
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

        self.dont_dump_for_first_n_it = dont_save_for_first_n_it
        self.n_times_checked = 0

    def condition_virtual(self, freq_cond, batch_id, epoch_id):
        if not freq_cond:
            return False

        # Check if dont_save_for_first_n_it has passed
        self.n_times_checked += 1

        # Check if the validation monitor has indeed recorded values
        if not self.validation_monitor.history:
            raise Exception('Best network saver should be placed after the'
                            'validation monitor in the list of extensions'
                            'provided to the Trainer object.')

        current_value = self.validation_monitor.history[-1][self.metric_idx]
        if self.m * current_value < self.m * self.best_value:
            self.best_value = current_value
            self.best_params_values = [p.get_value() for p in self.params]
            # Check if dont_save_for_first_n_it has passed
            if self.n_times_checked < self.dont_dump_for_first_n_it:
                return False  # we don't dump
            return True and self.save_on_disk  # we dump

        elif self.dont_dump_for_first_n_it == self.n_times_checked:
            return True and self.save_on_disk

        return False

    def compute_object(self):
        return (self.best_params_values,
                ['Best Network dumped into {}'.format(self.folder_path)])

    def restore(self, file_path=None):
        if not file_path:
            file_path = os.path.join(self.folder_path, self.file_name + '.pkl')
        file_handle = open(file_path, 'r')
        self.best_params_values = cPickle.load(file_handle)
        file_handle.close()

        for p, v in zip(self.params, self.best_params_values):
            p.set_value(v)

    def finish(self, batch_id, epoch_id):
        b = time.time()
        msg = []

        # The network has not yet been saved on the disk
        if self.n_times_checked < self.dont_dump_for_first_n_it:
            self.dont_dump_for_first_n_it = 0.1
            self.save()
            msg.append('Best Network dumped into {}'.format(self.folder_path))

        if self.restore_best_weights_at_the_end:
            msg.append('... best network re-loaded')
            for p, v in zip(self.params, self.best_params_values):
                p.set_value(v)

        e = time.time()

        return e - b, msg


class MetricSaver(Saver):
    """Saves the history of a ValidationMonitor extension
    """

    def __init__(self, metric_monitor, freq, folder_path, name=None):
        if not name:
            name = metric_monitor.name_extension
        super(MetricSaver, self).__init__('Metric saver ' + name, freq,
                                          folder_path, 'metric_saver_' + name)
        self.metric_monitor = metric_monitor

    def compute_object(self):
        np_history = np.array(self.metric_monitor.history)
        np_iterations = np.array(self.metric_monitor.iterations)
        d = {'iterations': np_iterations,
             'history': np_history,
             'names': self.metric_monitor.metric_names}
        return (d,
                [self.name_extension +
                 ' Metric histories dumped into {}'.format(
                     self.folder_path)])


VariableSaver = MetricSaver


class ResetParams(Extension):
    def __init__(self, freq, params, monitor, metric_name, values_to_avoid,
                 idx=0):
        Extension.__init__(self, name_extension='ResetParams', freq=freq)
        self.freq = freq
        self.params = params
        self.monitor = monitor
        self.init_param_values = [p.get_value() for p in params]

        if not isinstance(values_to_avoid, list):
            values_to_avoid = [values_to_avoid]
        self.values_to_avoid = values_to_avoid

        self.metric = monitor.find_metric_from_name(metric_name)
        # Index of the metric to check in the monitoring extension
        self.metric_idx = monitor.output_links[self.metric][idx]

    def condition_virtual(self, freq_cond, batch_id, epoch_id):
        """The extension might only be run if certain conditions are met.
        """
        if not freq_cond:
            return False

        current_value = self.monitor.history[-1][self.metric_idx]
        return current_value in self.values_to_avoid

    def execute_virtual(self, batch_id, epoch_id=None):
        param_values = [np.random.permutation(p.flat).reshape(p.shape)
                        for p in self.init_param_values]

        for p, p_np in zip(self.params, param_values):
            p.set_value(p_np)

        return ['Weights have been reset']


# class SharedVariableSchedule(Extension, EndCondition):
#     """
#     Both extension and ending condition that decreases the learning rate if
#     there is no improvement on a metric monitored by a monitoring extension.
#     If does not improve for absolute_patience, then the training stops.
#
#     The frequence of this extension is the same as the :class:`Monitor` monitor
#     parameter.
#
#     Parameters:
#     -----------
#     monitor: :class:`Monitor` object
#         :class:`Monitor` object which computes the metrics you are
#         interested in.
#     metric: MonitoredQuantity or tensor
#         the metric (either MonitoredQuantity object or tensor) that
#         you are interested in.
#     idx: int, default=0
#         if metric computes several outputs, this index selects a single one.
#     learning_rate: theano shared variable
#         the variable storing the current learning rate
#     patience: int, default=5
#         the number of times we allow the metric to not improve before
#         decreasing the learning rate
#     max_patience: int, default=7
#         the number of times we allow the metric to not improve before we
#         stop the training.
#     decay_rate: float, default=2.0
#         the rate at which the learning rate is decreased
#     min_value: float
#         the minimal value that we tolerate for the learning rate. Below it, we
#         stop training.
#     params: list of shared variables (default None)
#         if you want the best parameters to be saved and restored. If you want
#         to include both the parameters of the network and those of the
#         optimisation algorithm (such as momentum), you may want to give
#         params=list(updates.keys()) as input.
#     """
#
#     def __init__(self, monitor, metric_name, learning_rate, idx=0, patience=5,
#                  max_patience=7, decay_rate=2., min_value=1e-12, params=None,
#                  metric_mode='min'):
#         Extension.__init__(self, 'Learning rate', monitor.freq)
#         EndCondition.__init__(self, 'Learning rate', monitor.freq)
#         self.lr = learning_rate
#         self.patience = patience
#         self.absolute_patience = max_patience
#         self.decay_rate = decay_rate
#         self.waiting = 0
#         self.absolute_waiting = 0
#         self.validation_monitor = monitor
#         self.min_value = min_value
#
#         self.params = params
#         self.best_params = [p.get_value() for p in self.params]
#
#         self.metric_name = metric_name
#         self.metric = monitor.find_metric_from_name(metric_name)
#         # Index of the metric to check in the monitoring extension
#         self.metric_idx = monitor.output_links[self.metric][idx]
#         self.mode_metric = metric_mode
#         if metric_mode == 'max':
#             self.m = -1
#         elif metric_mode == 'min':
#             self.m = 1
#         else:
#             raise ValueError
#
#         self.best_value = self.m * np.inf
#
#     def execute_virtual(self, batch_id, epoch_id):
#         current_value = self.validation_monitor.history[-1][self.metric_idx]
#         if np.isnan(current_value):
#             raise Exception('nan detected')
#
#         if current_value * self.m < self.best_value * self.m:
#             self.best_value = current_value
#             self.waiting = 0
#             self.absolute_waiting = 0
#             if self.params:
#                 self.best_params = [p.get_value() for p in self.params]
#         else:
#             self.waiting += 1
#             self.absolute_waiting += 1
#
#         strs = ['Learning rate: {}, waiting {}/{}, absolute waiting {}/{}, '
#                 'best {} = {}'.format(
#             self.lr.get_value(), self.waiting, self.patience,
#             self.absolute_waiting, self.absolute_patience, self.metric_name,
#             self.best_value)]
#
#         if self.waiting > self.patience:
#             self.lr.set_value(self.lr.get_value() / self.decay_rate)
#             self.waiting = 0
#             msg = 'Learning rate decreased'
#             if self.params:
#                 for p, v in zip(self.params, self.best_params):
#                     p.set_value(v)
#                 msg += '... best network re-loaded'
#             strs.append(msg)
#         return strs
#
#     def check_condition_virtual(self, batch_id, epoch_id):
#         res = False
#         if self.absolute_waiting > self.absolute_patience:
#             res = 'Patience exceeded'
#         elif self.lr.get_value() < self.min_value:
#             res = 'Learning rate too small'
#
#         if res and self.params:
#             for p, v in zip(self.params, self.best_params):
#                 p.set_value(v)
#             res += '... best network re-loaded'
#
#         if res:
#             res = [res]
#         return res
