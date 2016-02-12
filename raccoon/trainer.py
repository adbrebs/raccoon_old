import sys
import time

from utils import print_wrap


class Trainer:
    def __init__(self, train_monitor, extensions, end_conditions):
        self.train_monitor = train_monitor
        self.extensions = [train_monitor] + extensions
        self.end_conditions = end_conditions
        self.total_time = 0

    def print_extensions_logs(self, extensions_logs):
        for ext, (timing, logs) in extensions_logs:
            print print_wrap(
                '{} [{:.3g} sec]:'.format(ext.name_extension, timing), 1)
            for line in logs:
                print print_wrap(line, 2)

    def print_end_conditions_logs(self, cond_logs):
        for cond, logs in cond_logs:
            print print_wrap(
                '{}:'.format(cond.name), 1)
            for line in logs:
                print print_wrap(line, 2)

    def process_batch(self, epoch, iteration, *inputs):
        if iteration == 0:
            return self.start()

        self.train_monitor.train(*inputs)

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

    def start(self):
        self.total_time = time.clock()
        print '\nTraining starts!'
        print '    Computing potential initial extensions...',
        extensions_logs = [(ext, ext.start())
                           for ext in self.extensions if ext.apply_at_the_start]
        print 'Done!'
        if extensions_logs:
            print 'Before training (extensions that run at the start):'
        self.print_extensions_logs(extensions_logs)

    def finish(self, batch_id):
        self.total_time = time.clock() - self.total_time
        print 'Training finished after {} seconds'.format(self.total_time)
        print 'Computing extensions...',
        extensions_logs = [(ext, ext.finish(batch_id))
                           for ext in self.extensions if ext.apply_at_the_end]
        print 'Done!'
        self.print_extensions_logs(extensions_logs)
