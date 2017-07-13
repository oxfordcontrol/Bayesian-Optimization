import GPy
import numpy as np
import os
from methods import QEI, QEI_CL, OEI, LP_EI, Random_EI, Caller
from test_functions import gp
import matplotlib.pyplot as plt


def plot2d(fun, bounds, N, contour=True, contourf=True, fig=None, ax=None):
    x = np.linspace(bounds[0][0], bounds[0][1], N)
    y = np.linspace(bounds[1][0], bounds[1][1], N)

    X, Y = np.meshgrid(x, y)

    Z = np.zeros(X.shape)
    for i in range(0, X.shape[0]):
        for j in range(0, X.shape[1]):
            Z[i, j] = fun(np.array([[X[i, j], Y[i, j]]]))[0]

    if fig is None or ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    if contourf:
        heatmap = ax.contourf(X, Y, Z, 20, cmap=plt.get_cmap('Greys'))
    else:
        heatmap = None

    if contour:
        ax.contour(X, Y, Z, 30, colors='k')

    return fig, ax, heatmap


def define_experiments():
    experiments = []

    ####################
    # Test draws of a Gaussian Process
    ####################
    input_dim = 2
    lengthscale = np.array([1])/4
    bounds = np.repeat([[0., 1.]], input_dim, axis=0)
    kernel = GPy.kern.RBF(input_dim=input_dim, variance=0.3,
                          lengthscale=lengthscale, ARD=True)
    objective = gp(kernel=kernel.copy(), bounds=bounds, sd=1e-6)

    options = {'batch_size': 2,
               'iterations': 1,
               'objective': objective,
               'kernel': kernel,
               'gp_opt_restarts': 0,  # Leave gp model as it is
               'acq_opt_restarts': 100,
               'initial_size': 10,
               'normalize_Y': False,
               'noiseless': False,
               'noise': 1e-6,
               'liar_choice': 'max'
               }

    seeds = np.arange(100, 100 + 1000)

    experiments.append(Caller('Actual EI', QEI(options)))
    experiments.append(Caller('Optimistic EI', OEI(options)))
    experiments.append(Caller('LP Heuristic', LP_EI(options)))
    experiments.append(Caller('CL Heuristic', QEI_CL(options)))
    experiments.append(Caller('Random Heuristic', Random_EI(options)))

    return experiments, seeds


def run_experiments():
    # This prevents libraries from using all available threads
    os.environ["OMP_NUM_THREADS"] = "1"

    ''' Run the experiments '''
    num_threads = 3
    experiments, seeds = define_experiments()
    for i in range(0, len(experiments)):
        print('-------------')
        print(experiments[i].job_name)
        print('-------------')
        experiments[i].run_multiple(seeds=seeds, num_threads=num_threads)
        experiments[i].save_data()

    '''
    Create a detailed plot, with the GP mean, GP sd, and the algorithms'
    choices for the first seed.
    '''
    # Build GP model
    X0 = experiments[0].points[0][0:10]
    y0 = experiments[0].outputs[0][0:10]
    m = GPy.models.GPRegression(X0, y0, experiments[0].bo.kernel)
    m.Gaussian_noise.constrain_fixed(experiments[0].bo.noise)

    bounds = experiments[0].bo.bounds  # + np.array([-.1, .1])
    # Plot GP's mean value
    fig, ax, heatmap = plot2d(
        lambda x: m.predict_noiseless(x)[0],
        bounds, 100,
        contour=True, contourf=False)

    fig, ax, heatmap = plot2d(
        lambda x: np.sqrt(m.predict_noiseless(x)[1]),
        bounds, 100,
        contour=False, contourf=True,
        fig=fig, ax=ax)

    # qEI's choice
    X = experiments[0].points[0][10:12]
    ax.plot(X[:, 0], X[:, 1], 'ko', markersize=7,
            markerfacecolor='none', label='qEI')
    # oEI's choice
    X = experiments[1].points[0][10:12]
    ax.plot(X[:, 0], X[:, 1], 'kx', markersize=7, label='oEI')
    # LP's choice
    X = experiments[2].points[0][10:12]
    ax.plot(X[:, 0], X[:, 1], 'k+', markersize=7, label='LP')
    plt.legend()
    plt.show()

    '''
    Calculate the actual expected improvements of all the algorithms
    and print their mean difference as compared to the optimal strategy (qEI)
    '''
    ei = []
    for i in range(0, len(experiments)):
        ei.append(np.zeros(len(seeds)))
        for j in range(len(seeds)):
            # Build GP model
            X0 = experiments[i].points[j][0:10]
            y0 = experiments[i].outputs[j][0:10]
            m = GPy.models.GPRegression(X0, y0, experiments[0].bo.kernel)
            m.Gaussian_noise.constrain_fixed(experiments[0].bo.noise)

            # Get algorithm's choice
            X = experiments[i].points[j][10:12]
            '''
            Calculate the actual expected improvement. Although qEI can
            sometimes be arbitrarily wrong (see methods/qEI_problem.R), the
            expectation here is low dimensional and the values returned by qEI
            were found to be accurate as compared to calculating the
            expectation by sampling
            '''
            ei[i][j], _ = QEI.acquisition_fun(X, m)

    print('-------------------')
    print('Achieved improvements')
    print('-------------------')
    for i in range(0, len(experiments)):
        print(experiments[i].job_name, np.mean(ei[i]))

    print('-------------------')
    print('% Mean Difference from qEI')
    print('-------------------')
    for i in range(0, len(experiments)):
        if i != 0:
            print(experiments[i].job_name,
                  np.mean(ei[0] - ei[i]) /
                  np.mean(ei[0]) * 100)

    return ei


if __name__ == "__main__":
    ei = run_experiments()
