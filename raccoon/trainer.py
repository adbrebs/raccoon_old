import sys
import time

from utils import print_wrap


class Trainer:
    def __init__(self, train_monitor, data_generator, extensions,
                 end_conditions):
        self.train_monitor = train_monitor
        self.extensions = [train_monitor] + extensions
        self.end_conditions = end_conditions
        self.data_generator = data_generator
        self.iteration = self.epoch = self.begin_time = 0
        self.data_processing_time = 0

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

    def train(self, custom_process_fun=None):
        if self.iteration == 0:
            self.start()

        try:
            while True:
                self.epoch += 1
                epoch_iterator = self.data_generator()

                while True:
                    t = time.clock()
                    inputs = next(epoch_iterator, None)
                    self.data_processing_time += time.clock() - t
                    if not inputs:
                        break

                    self.iteration += 1

                    if custom_process_fun:
                        res = custom_process_fun(inputs)
                    else:
                        res = self.process_batch(*inputs)

                    if res:
                        self.finish()
                        sys.exit()
        except KeyboardInterrupt:
            print 'Training interrupted by user.'
            self.finish()

    def process_batch(self, *inputs):
        """
        Returns True if an ending condition triggers
        """
        self.train_monitor.train(*inputs)

        extensions_logs = [(ext, ext.execute(self.iteration))
                           for ext in self.extensions
                           if ext.check(self.iteration)]
        cond_logs = []
        for cond in self.end_conditions:
            logs = cond.check_condition(self.iteration)
            if logs:
                cond_logs.append((cond, logs))

        # If no extensions are active
        if not any(extensions_logs) and not any(cond_logs):
            return False

        print 'Epoch {}, iteration {}, spent time {:.3f} secs:'.format(
            self.epoch, self.iteration, time.clock()-self.begin_time)
        self.print_extensions_logs(extensions_logs)
        self.print_end_conditions_logs(cond_logs)
        print '-'*79
        sys.stdout.flush()  # Important if output is redirected

        if any(cond_logs):
            return True
        return False

    def start(self):
        self.begin_time = time.clock()
        print '\nTraining starts!'
        sys.stdout.flush()
        print print_wrap('Computing potential initial extensions...', 1),
        extensions_logs = [(ext, ext.start())
                           for ext in self.extensions
                           if ext.apply_at_the_start]
        print 'Done!'
        if extensions_logs:
            print 'Before training (extensions that run at the start):'
        self.print_extensions_logs(extensions_logs)

    def finish(self):
        time_spent = time.clock() - self.begin_time
        print 'Training finished after {} seconds'.format(time_spent)
        print 'Computing extensions...',
        extensions_logs = [(ext, ext.finish(self.iteration))
                           for ext in self.extensions if ext.apply_at_the_end]
        print 'Done!'
        self.print_extensions_logs(extensions_logs)

        # Total extension time
        total_ext_time = 0
        for ext in self.extensions[1:]:  # remove training
            total_ext_time += ext.total_spent_time_in_ext

        logs = [
            (0, 'Data processing', self.data_processing_time),
            (0, 'Training', self.train_monitor.total_spent_time_in_ext),
            (0, 'Extensions', total_ext_time)]

        time_recorded = (self.data_processing_time +
                         self.train_monitor.total_spent_time_in_ext)
        for ext in self.extensions[1:]:  # remove training
            logs.append((1, ext.name_extension, ext.total_spent_time_in_ext))
            time_recorded += ext.total_spent_time_in_ext

        total_time = time.clock() - self.begin_time
        logs.append((0, 'Overhead training loop',
                     self.total_time - time_recorded))

        print '\nProfiling: (Total time: {:.3f} secs)'.format(total_time)
        for level, name, timing in logs:
            print print_wrap('[{:.3f} %] ({:.3f} secs) : {}'.format(
                100.0 * timing / total_time, timing, name), 1 + level)
