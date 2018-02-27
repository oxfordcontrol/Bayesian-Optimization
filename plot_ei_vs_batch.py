from __future__ import print_function
import numpy as np
import sys
import glob
import os
import matplotlib.pyplot as plt
# When no X server is present
plt.switch_backend('agg')

'''
This script prints and plots the averaged expected improvement results for
gp posteriors as calculated and saved by gp_posteriors.py in the folders
specified by the parameters used to call the script
'''


def print_results(folders):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for k in range(len(folders)):
        folder = folders[k]

        print('------', folder, '------')
        files = glob.glob(folder + '/*.npz')
        results = None
        results_sampling = None
        for file in files:
            npzfile = np.load(file)

            if results is not None:
                results = np.concatenate((results, npzfile['results']), axis=1)
                results_sampling = np.concatenate(
                    (results_sampling, npzfile['results_sampling']), axis=1
                    )
            else:
                results = npzfile['results']
                results_sampling = npzfile['results_sampling']

        for i in range(results.shape[0]):
            print('->Batch size:', i + 1)
            print('Mean qEI:', '%.3f' % np.mean(results[i]))
            print('Mean (sampled) EI:', '%.3f' % np.mean(results_sampling[i]))

        colors = ['r', 'g', 'b', 'y', 'c', 'm']
        label = '-'.join(os.path.basename(folder).split('_')[1:])
        ax.plot(np.arange(results.shape[0]) + 1,
                np.mean(results_sampling, axis=1),
                color=colors[k], marker='s', label=label, linewidth=2)
        ax.legend()
        ax.set_xlabel('Batch size')
        ax.set_ylabel('Averaged Expected Improvement')
        fig.set_size_inches((4.5, 4))
        fig.tight_layout()
        plt.savefig('results/EI_vs_batch_size.pdf')

    plt.show()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        folders = sys.argv[1:]
    else:
        print('Please provide the folders of the experiments')
        assert False

    print_results(folders=folders)
