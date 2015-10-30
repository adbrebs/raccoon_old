import sys

from utils import print_wrap


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

    def process_batch(self, epoch, iteration, *inputs):
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
