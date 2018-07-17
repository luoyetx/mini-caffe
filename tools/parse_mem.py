#!/usr/bin/env python2.7
# pylint: disable=invalid-name, no-member, line-too-long

import os
import re
import argparse
from collections import namedtuple
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

sns.set()

Entry = namedtuple('Entry', ('type', 'device', 'time', 's1', 's2'))

class MemStatistic(object):
    """memory statistic

    A Memory Entry (type, device, time, size1, size2)
    For hit request: ('hit', device, time, has, wants)
    For miss request: ('miss', device, time, creates, wants), creates == wants
    For return: ('return', device, time, rets, rets)
    For free: ('free', device, time, freed, freed)
    """

    def __init__(self):
        self.entry_ = []
        self.clock_ = 0

    def request_hit(self, dev, has, wants):
        """request hit the memory pool and not create any extra memory
        """
        self.entry_.append(Entry('hit', dev, self.clock_, has, wants))
        self._clock_update()

    def request_miss(self, dev, creates, wants):
        """request miss the memory pool and create an extra memory
        """
        self.entry_.append(Entry('miss', dev, self.clock_, creates, wants))
        self._clock_update()

    def return_mem(self, dev, size):
        """return a memory entry to the pool
        """
        self.entry_.append(Entry('return', dev, self.clock_, size, size))
        self._clock_update()

    def free_mem(self, dev, size):
        """free memory in pool
        """
        self.entry_.append(Entry('free', dev, self.clock_, size, size))
        self._clock_update()

    def _clock_update(self):
        """update internal clock
        """
        self.clock_ += 1

    def plot(self):
        """plot statistic
        """
        cpu_entry = list(filter(lambda e: e.device == 'cpu', self.entry_))
        gpu_entry = list(filter(lambda e: e.device == 'gpu', self.entry_))
        n = len(self.entry_)
        x_time = np.arange(n + 1)

        def entry_analyse(entrys):
            # entry analyse
            y_total = np.zeros(n + 1)
            y_unused = np.zeros(n + 1)
            for entry in entrys:
                t = entry.time
                if entry.type == 'hit':
                    y_unused[t] = -entry.s1
                elif entry.type == 'miss':
                    y_total[t] = entry.s1
                elif entry.type == 'return':
                    y_unused[t] = entry.s1
                elif entry.type == 'free':
                    y_total[t] = -entry.s1
                    y_unused[t] = -entry.s1
                else:
                    raise ValueError('No such entry type')
            # scan
            y_total = np.cumsum(y_total)
            y_unused = np.cumsum(y_unused)
            # convert to M
            y_total /= 1024 * 1024
            y_unused /= 1024 * 1024
            return y_total, y_unused

        y_cpu_total, y_cpu_unused = entry_analyse(cpu_entry)
        y_gpu_total, y_gpu_unused = entry_analyse(gpu_entry)
        # plot over time
        plt.xlabel('time / tick')
        plt.ylabel('memory / MB')
        plt.plot(x_time, y_cpu_total - y_cpu_unused, 'b', label='[cpu] used')
        plt.plot(x_time, y_cpu_total, 'r', label='[cpu] all')
        plt.plot(x_time, y_gpu_total - y_gpu_unused, 'c', label='[gpu] used')
        plt.plot(x_time, y_gpu_total, 'y', label='[gpu] all')
        plt.legend(loc='upper left')
        plt.show()


def parse_args():
    """parse command argument
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help="memory statistic result file")
    args = parser.parse_args()
    assert os.path.exists(args.input), "input file not exists"
    return args

def convert(s, t):
    """convert memory size of type
    """
    assert t in 'BKM'
    if t == 'B':
        return int(s)
    elif t == 'K':
        return int(s * 1024)
    else:
        return int(s * 1024 * 1024)

if __name__ == '__main__':
    args = parse_args()
    statistic = MemStatistic()
    with open(args.input, 'r') as fin:
        for i, line in enumerate(fin):
            if '[CPU]' in line or '[GPU]' in line:
                dev = 'cpu' if '[CPU]' in line else 'gpu'
                if 'Requested' in line:
                    if 'Get' in line:
                        # hit
                        g = re.search(r'Requested (.+) ([B|K|M]), Get (.+) ([B|K|M])', line).groups()
                        assert len(g) == 4
                        wants = convert(float(g[0]), g[1])
                        has = convert(float(g[2]), g[3])
                        statistic.request_hit(dev, has, wants)
                    else:
                        # miss
                        assert 'Create' in line
                        g = re.search(r'Requested (.+) ([B|K|M]), Create (.+) ([B|K|M])', line).groups()
                        assert len(g) == 4
                        wants = convert(float(g[0]), g[1])
                        creates = convert(float(g[2]), g[3])
                        assert wants == creates
                        statistic.request_miss(dev, creates, wants)
                elif 'Return' in line:
                    # return memory
                    g = re.search(r'Return (.+) ([B|K|M])', line).groups()
                    assert len(g) == 2
                    rets = convert(float(g[0]), g[1])
                    statistic.return_mem(dev, rets)
                elif 'Free' in line:
                    # free memory
                    g = re.search(r'Free (.+) ([B|K|M])', line).groups()
                    assert len(g) == 2
                    rets = convert(float(g[0]), g[1])
                    statistic.free_mem(dev, rets)
                else:
                    # something else, ignore
                    pass
    # plot
    statistic.plot()
