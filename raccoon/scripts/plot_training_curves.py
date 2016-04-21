#!/usr/bin/env python

import argparse
import cPickle
from itertools import cycle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument('feature')
parser.add_argument('files', nargs='+')

options = parser.parse_args()

files = options.files
feature_name = options.feature


fig = plt.figure(num=None, figsize=(14, 9), dpi=80, facecolor='w', edgecolor='k')
cmap = plt.get_cmap('prism')
# colors = [cmap(i) for i in np.linspace(0, 1, 10*len(FOLDERS))]
colors = ['b', 'g', 'r', 'm', 'y', 'k', 'c']
lines = ["-","--","-.",":"]
linecycler = cycle(lines)
colorcyler = cycle(colors)

for f in files:
    data = cPickle.load(open(f, 'r'))
    c = next(colorcyler)
    t = next(linecycler)

    names = data['names']
    if feature_name in names:
        id_feature = names.index(feature_name)
    else:
        raise Exception('Feature note found. Possible features are {}'.format(
            names))

    feature = data['history'][:, id_feature]

    plt.plot(data['iterations'], feature,
             '.' + t, color=c, label=f + '_' + feature_name)

plt.xlabel('Number of minibatches')
plt.ylabel(feature_name)
plt.legend(loc='best')
plt.title(feature_name)
plt.savefig('plot.pdf')

