from methods import Caller
import os.path
import matplotlib.pyplot as plt
import sys
'''
This script plots the results of various algorithms at the same plot, and
saves it as a pdf in the results/pdf folder.

Call it as:
python plot_experiments filename1 ... filenameN
where filename* are the filenames of the saved experiments.
'''


def plot_data(filenames, plot_random=True, groups=None):
    # Load experiments
    experiments = []
    for filename in filenames:
        if os.path.isfile(filename):
            experiments.append(Caller(filename=filename))
        else:
            print('File:', filename, 'not found.')
            assert False

    # Plotting
    # Hopefully we won't plot more than 6 different algorithms at the same plot
    colors = ['b', 'r', 'g', 'c', 'm', 'y']
    fig = None
    ax = None
    offset = -0.15
    for k in range(len(experiments)):
        job_name = experiments[k].job_name
        label = job_name.split('_')[-1]
        fig, ax = experiments[k].plot(color=colors[k], fig=fig,
                                      ax=ax, label=label,
                                      offset=offset)
        offset += 0.1

    if plot_random:
        fig, ax = experiments[0].plot_random(color='k', fig=fig,
                                             ax=ax, offset=offset,
                                             label='random')

    function_name = job_name.split('_')[0]
    ax.set_xlabel('Number of Batches')
    ax.set_ylabel('Min Function Value')
    ax.set_title(function_name)
    figure = ax.get_figure()
    figure.set_size_inches((4, 3))

    # Save plot
    name = os.path.basename(filenames[0])
    name = os.path.splitext(name)[0]
    plt.tight_layout()
    plt.savefig('results/pdfs/' + name + '.pdf')


if __name__ == '__main__':
    if len(sys.argv) > 1:
        filenames = sys.argv[1:]
    else:
        print('Please provide the filenames of the results experiments')
        assert False

    plot_data(filenames=filenames, plot_random=True)
