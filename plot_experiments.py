import os.path
import glob
import numpy as np
import pickle
import argparse
import matplotlib

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

matplotlib.rc('font', size=SMALL_SIZE)          # controls default text sizes
matplotlib.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
matplotlib.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
matplotlib.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
matplotlib.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
matplotlib.rc('figure', titlesize=SMALL_SIZE)  # fontsize of the figure title

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
# When no X server is present
plt.switch_backend('agg')

'''
This script plots the results of various algorithms at the same plot, and
saves it as a pdf in the results/folder.

Call it as:
python plot_experiments fig_name folder1 ... folderN
where folder* are the folders of the saved experiments.
'''


def plot_mins(mins, options, color='b', fig=None, ax=None, label=None, offset=0):
    # Auxiliary function used by plot(), plot_random()
    if fig is None or ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    ax.ticklabel_format(style='sci', scilimits=(0, 1.5))
    for i in range(0, mins.shape[1], options.step):
        ax.scatter(i + 0*mins[:, i] + offset, mins[:, i],
                   s=50, marker='.', color=color, edgecolor='none',
                   alpha=0.3)
        ax.scatter(i + offset, np.median(mins[:, i]),
                   s=20, marker='d', color=color, edgecolor=(0, 0, 0))

    # ax.legend().get_frame().set_facecolor('none')
    # plt.legend(frameon=False)
    max_plotted_value = np.max(np.percentile(mins, 100, axis=0))
    if options.regret:
        ax.set_ylim(0, 1.05*max_plotted_value)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    return fig, ax


def plot(outputs, fmin, iterations, initial_size, batch_size, label, options,
         color='b', fig=None, ax=None, output_idx=0, offset=0):
    '''
    Plot error bars for the minimum achieved at each iteration of the BO
    Beware of the following inputs:
        output_idx: the output of the function bo.objective.f() might
                    have multiple collumns. The first one always contains
                    the output that is visible to the BO. Other collumns
                    contain auxiliary data, e.g. accuracy on the test set
                    etc. Hence one might want to plot the performance with
                    regards to these auxiliary data.
        offset: the plot is essentially a group of vertical lines. When
                plotting multiple results in the same figure, we shift the
                horizontal location of the lines by adding an offset to
                them.
    '''
    n = len(outputs)
    mins = np.zeros((n, iterations + 1))
    for i in range(n):
        for j in range(iterations + 1):
            idx = np.argmin(outputs[i][0:initial_size + j*batch_size, 0])
            mins[i, j] = outputs[i][idx, output_idx] - fmin

    fig, ax = plot_mins(mins, options, color, fig, ax, label, offset)

    return fig, ax


def plot_experiments(options):
    # Hopefully we won't plot more than 6 different algorithms at the same plot
    colors = ['r', 'b', 'g', 'y', 'b', 'm']
    fig = None
    ax = None
    offset = options.offset_start
    for k in range(len(options.folders)):
        folder = options.folders[k]
        # Load command line arguments
        with open(folder + '/arguments.pkl', 'rb') as file:
                args = pickle.load(file)
        # print(args)
        fmin = np.loadtxt(folder + '/fmin.txt')

        fails = 0
        outputs = []
        files = glob.glob(folder + '/*.npz')
        for file in files:
            npzfile = np.load(file)

            if npzfile['Y'].shape != ():
                outputs.append(npzfile['Y'])
            else:
                fails = fails + 1
                print(file)

        label = os.path.basename(folder).split('_')[1]

        if fails > 0:
            print(label, 'Fails:', fails,
                  'Successes: ', len(files) - fails)

        if k == len(options.folders) - 1:
            color = 'k'
        else:
            color = colors[k]

        fig, ax = plot(outputs=outputs,
                       fmin=fmin,
                       iterations=args.iterations,
                       initial_size=args.initial_size, # + args.init_replicates,
                       batch_size=args.batch_size,
                       color=color, fig=fig,
                       ax=ax, label=label,
                       offset=offset, options=options)
        offset += options.offset_delta

    ax.set_xlabel('Number of Batches')
    if options.regret:
        ax.set_ylabel('Regret')
    else:
        ax.set_ylabel('Loss')
    ax.set_title(options.name[0])
    figure = ax.get_figure()
    figure.set_size_inches((options.sizex, options.sizey))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    ax.locator_params(nbins=4, axis='y')

    # Save plot
    plt.tight_layout()
    plt.savefig('results/' + options.name[0] + '.pdf')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("name", nargs=1)
    parser.add_argument("folders", nargs="+")
    parser.add_argument('--linewidth', type=float, default=1)
    parser.add_argument('--capsize', type=float, default=1.5)
    parser.add_argument('--offset_start', type=float, default=-0.2)
    parser.add_argument('--offset_delta', type=float, default=0.1)
    parser.add_argument('--sizex', type=float, default=5)
    parser.add_argument('--sizey', type=float, default=2.5)
    parser.add_argument('--regret', type=int, default=1)
    parser.add_argument('--step', type=int, default=1)
    args = parser.parse_args()

    plot_experiments(args)
