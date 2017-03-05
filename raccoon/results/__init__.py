import cPickle
import os
from itertools import cycle
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pylab as plt
# matplotlib.rcParams['ps.useafm'] = True
# matplotlib.rcParams['pdf.use14corefonts'] = True
# matplotlib.rcParams['text.usetex'] = True
# matplotlib.rcParams['font.family'] = 'serif'
# matplotlib.rcParams['font.serif'] = 'cm'




def plot_training_curves(
        folders, metric_name='mse', sets=['valid', 'train'],
        print_set_in_legend=True):

    fig = plt.figure(num=None, figsize=(9, 5), dpi=80, facecolor='w',
                     edgecolor='k')
    plt.get_current_fig_manager().window.raise_()
    cmap = plt.get_cmap('prism')
    # colors = [cmap(i) for i in np.linspace(0, 1, 10*len(FOLDERS))]
    colors = ['b', 'g', 'r', 'm', 'y', 'k', 'c']
    lines = ["-","--","-.",":"]
    colorcyler = cycle(colors)

    for (folder, name) in folders:
        c = next(colorcyler)
        for i, set in enumerate(sets):
            f = os.path.join(folder, 'metric_saver_' + set + '.pkl')
            data = cPickle.load(open(f, 'r'))

            ls_metric_names = data['names']
            if metric_name in ls_metric_names:
                id_feature = ls_metric_names.index(metric_name)
            else:
                continue
            if not data['history'].any():
                continue
            feature = data['history'][:, id_feature]

            if print_set_in_legend:
                label = set + '_' + name
            else:
                label = name
            plt.plot(data['iterations'], feature,
                     lines[i], color=c, label=label)

    plt.xlabel('Number of minibatches')
    plt.ylabel(metric_name)
    plt.legend(loc='best', frameon=False,
               prop={'size':13})
    plt.title('Validation ' + metric_name +' through training')
    plt.grid(True, alpha=0.5)

    # fig.legend(legend_curves, legend_names, 'lower center', ncol=2, frameon=False,
    #            prop={'size':12})
    # plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.6, hspace=3)
    plt.show()
    # plt.savefig('comparison_attention.pdf')