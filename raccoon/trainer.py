import sys
import time

from utils import print_wrap


class Trainer:
    def __init__(self, train_monitor, extensions, end_conditions,
                 run_before_training=False):
        self.train_monitor = train_monitor
        self.extensions = [train_monitor] + extensions
        self.end_conditions = end_conditions
        self.run_before_training = run_before_training
        self.total_time = 0

    def print_extensions_logs(self, extensions_logs):
        for ext, logs in extensions_logs:
            print print_wrap(
                '{}:'.format(ext.name_extension), 1)
            for line in logs:
                print print_wrap(line, 2)

    def print_end_conditions_logs(self, cond_logs):
        for cond, logs in cond_logs:
            print print_wrap(
                '{}:'.format(cond.name), 1)
            for line in logs:
                print print_wrap(line, 2)

    def process_batch(self, epoch, iteration, *inputs):
        self.train_monitor.train(*inputs)

        if iteration == 0:
            self.total_time = time.clock()
            if not self.run_before_training:
                return False

        extensions_logs = [(ext, ext.execute(iteration))
                           for ext in self.extensions if ext.check(iteration)]
        cond_logs = []
        for cond in self.end_conditions:
            logs = cond.check_condition(iteration)
            if logs:
                cond_logs.append((cond, logs))

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
        self.total_time = time.clock() - self.total_time
        print 'Training finished after {} seconds'.format(self.total_time)
        print 'Computing extensions...',
        print 'Done!'
        extensions_logs = [(ext, ext.execute(batch_id))
                           for ext in self.extensions if ext.apply_at_the_end]
        self.print_extensions_logs(extensions_logs)
