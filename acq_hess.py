import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random
import methods
from gp_posteriors import create_options, create_parser

# Default backend
plt.switch_backend('Qt5Agg')
# When no X server is present
# plt.switch_backend('agg')


def plot(bo):
    # x = np.linspace(-1, 1, 500)[:, None]
    x_min = 0.05
    x_max = 0.2
    x = np.linspace(x_min, x_max, 1000)[:, None]
    # Include also the observations
    x = np.sort(np.vstack((x, bo.X.value.copy())), axis=0)

    fig, axes = plt.subplots(4, 1, sharex=True)

    ax = axes[1]
    y = np.zeros(x.shape)
    for i in range(len(x)):
        y[i] = bo.acquisition(x[i])[0]
    ax.plot(x, y, 'k', lw=2)
    ax.set_ylabel('Acq')
    ax.set_xlabel('x')

    ax = axes[2]
    y = np.zeros(x.shape)
    for i in range(len(x)):
        y[i] = bo.acquisition(x[i])[1]
    ax.plot(x, y, 'k', lw=2)
    ax.set_ylabel('Grad')
    ax.set_xlabel('x')

    ax = axes[3]
    y = np.zeros(x.shape)
    for i in range(len(x)):
        y[i] = bo.acquisition_hessian(x[i])[0]
    ax.plot(x, y, 'k', lw=2)
    ax.set_ylabel('Hess')
    ax.set_xlabel('x')

    ax = axes[0]
    mean, var = bo.predict_y(x)
    ax.plot(bo.X.value, bo.Y.value, 'ko', mew=2)
    ax.plot(x, mean, 'b', lw=2)
    ax.fill_between(x[:, 0],
                    mean[:, 0] - 2*np.sqrt(var[:, 0]),
                    mean[:, 0] + 2*np.sqrt(var[:, 0]),
                    color='blue', alpha=0.2)
    ax.set_ylabel('Function')
    ax.axis((x_min, x_max, -25, 30))

    fig.set_size_inches((4, 8))
    fig.tight_layout()
    plt.savefig('results/acquisition.pdf')
    plt.show(block=True)


def main(args):
    k = 1
    options = create_options(args)[0]
    options['batch_size'] = k
    options['iterations'] = 0
    options['noise'] = 1e-2

    tf.reset_default_graph()
    tf.set_random_seed(options['seed'])
    np.random.seed(options['seed'])
    random.seed(options['seed'])

    bo = methods.Random_EI(options)

    bo.bayesian_optimization()
    plot(bo)


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args)