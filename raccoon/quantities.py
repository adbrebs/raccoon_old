import heapq
import theano.tensor as T


class MonitoredQuantity(object):
    """This abstract class allows to monitor quantities that are not theano
    tensors.
    If computed theano tensor values are required, they need to be indicated in
    required_tensors.

    If there are multiple quantities computed, name has to be a list with as
    many names as quantities returned by the calculate method.

    See KErrorRate for an example.

    Parameters:
    -----------
    name_or_names: str or list of strings
        names of the quantities being computed
    required_tensors: list of Theano tensors (default=None)
        if specified, the values of these tensors will be provided to the
    """
    def __init__(self, name_or_names, required_tensors=None):
        if not isinstance(name_or_names, list):
            name_or_names = [name_or_names]
        self.names = name_or_names
        self.n_outputs = len(name_or_names)
        if not required_tensors:
            required_tensors = []
        self.required_tensors = required_tensors

    @property
    def name(self):
        if len(self.names) > 1:
            raise ValueError('This monitored quantity has several variables')
        return self.names[0]

    @name.setter
    def name(self, new_name):
        if len(self.names) > 1:
            raise ValueError('This monitored quantity has several variables')
        self.names = [new_name]

    def calculate(self, *inputs):
        """Computes the outputs using, optionnally, values specified in
        required_tensors.

        Parameters
        ----------
        *inputs: list of numpy arrays
            the list of the computed tensor values returned by the trainer.
            The order of the inputs is the SAME as the order provided in
            required_tensors.

        Returns either a single value of a list of values
        """
        pass

    def write_str(self, strs, values, indent=''):
        """
        The way the results are printed by the :class:`Trainer` class.
        You can overwrite this method if you want a different display.

        Parameters
        ----------
        strs: list of strings
            the list of strings to write in. Each element represents a line to
            be printed.
        """
        for name, val in zip(self.names, values):
            strs.append(indent + name + ': {}'.format(val))


class LayerStatistic(MonitoredQuantity):
    """
    Display activations and gradients of the cost with respect to these
    activations.
    """
    def __init__(self, activations, cost):

        gradients = T.grad(cost, activations)
        act_mean = [out.mean() for out in activations]
        act_var = [out.std(axis=1).mean() for out in activations]
        grad_mean = [g.mean() for g in gradients]
        grad_var = [g.std(axis=1).mean() for g in gradients]
        self.n_layers = len(activations)

        out_names = []
        grad_names = []
        out_names += ['mean act {}'.format(i) for i in range(self.n_layers)]
        out_names += ['std act {}'.format(i) for i in range(self.n_layers)]
        grad_names += ['mean grad {}'.format(i) for i in range(self.n_layers)]
        grad_names += ['std grad {}'.format(i) for i in range(self.n_layers)]

        names = out_names + grad_names
        MonitoredQuantity.__init__(self, names,
                                   act_mean + act_var + grad_mean + grad_var)

    def calculate(self, *inputs):
        return inputs

    def write_str(self, strs, values, indent=''):
        """
        Prints on a single line
        """
        str_val = '(out, grad): '
        for i in range(self.n_layers):
            str_val += '([{:.3g}, {:.3g}], [{:.3g}, {:.3g}]) '.format(
                values[i], values[i+self.n_layers], values[i+2*self.n_layers],
            values[i+3*self.n_layers])
        strs.append(str_val)


class UpdateRatio(MonitoredQuantity):
    """
    Monitor the ratio update/parameter_value.
    """
    def __init__(self, updates):

        self.d_names = []
        ratios = []
        for a, b in updates.iteritems():
            ratios.append(T.abs_((a - b) / a).mean())
            self.d_names.append(a.name)

        names = ['ratio {}'.format(i) for i in range(len(ratios))]
        MonitoredQuantity.__init__(self, names, ratios)

    def calculate(self, *inputs):
        return inputs

    def write_str(self, strs, values, indent=''):
        """
        Prints on a single line
        """
        str_val = 'ratios: '
        for name, v in zip(self.d_names, values):
            str_val += '({}, {:.3g}) '.format(name, v)
        strs.append(str_val)


class KErrorRate(MonitoredQuantity):
    """
    Computes the k error rate for a classification model.

    Parameters
    ----------
    idx_target: theano 1D tensor of shape (batch_size,)
        the targets of a batch
    output: theano 2D tensor of shape (batch_size, output_size)
        the outputs of a batch
    k: int
        the k highest predictions to consider
    """
    def __init__(self, idx_target, output, k):
        name_or_names = '{} error rate'.format(k)
        MonitoredQuantity.__init__(self, name_or_names, [idx_target, output])
        self.k = k

    def calculate(self, idx_target, output):

        bs, out_dim = output.shape

        err_rate = 0
        for i in range(bs):
            idxs = heapq.nlargest(self.k, range(out_dim), output[i].take)
            if idx_target[i] not in idxs:
                err_rate += 1.0

        return err_rate / bs
