import os
import numpy as np
import argparse
import gpflow
import GPy
import GPyOpt
import methods
import time
import pickle
from test_functions import hart6, loghart6
import copy


class scale_function():
    def __init__(self, function):
        self.bounds = function.bounds.astype(float)
        self.function = function
        self.bounds[:, 0] = -1/2 
        self.bounds[:, 1] = 1/2 
        if hasattr(function, 'fmin'):
            self.fmin = function.fmin

    def f(self, X):
        means = (self.function.bounds[:, 1] + self.function.bounds[:, 0])/2
        lengths = self.function.bounds[:, 1] - self.function.bounds[:, 0]

        Xtrue = np.zeros(X.shape)
        for i in range(X.shape[0]):
            Xtrue[i, :] = X[i, :] * lengths + means

        return self.function.f(Xtrue)


def run(bo, seed, robust=True, save=False):
    start = time.time()
    try:
        X, Y = bo.bayesian_optimization(seed)
        end = time.time()
        print('Done with:', bo.options['job_name'], 'seed:', seed,
              'Time:', '%.2f' % ((end - start)/60), 'min')
    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt, stopping.")
        raise
    except:
        print('Experiment of', bo.options['job_name'],
              'with seed', seed, 'failed')
        X, Y = None, None
        if not robust:
            raise

    if save:
        save_folder = 'out/' + bo.options['job_name'] + '/'
        filepath = save_folder + str(seed) + '.npz'
        try:
            os.makedirs(save_folder)
        except OSError:
            pass
        try:
            os.remove(filepath)
        except OSError:
            pass

        np.savez(filepath, X=X, Y=Y)


def main(args):
    functions = {
        'branin': GPyOpt.objective_examples.experiments2d.branin(),
        'cosines': GPyOpt.objective_examples.experiments2d.cosines(),
        'sixhumpcamel': GPyOpt.objective_examples.experiments2d.sixhumpcamel(),
        'alpine1': GPyOpt.objective_examples.experimentsNd.alpine1(input_dim=5),
        'hart6': hart6(),
        'loghart6': loghart6()
    }

    algorithms = {
        'OEI': methods.OEI,
        'QEI': methods.QEI,
        'QEI_CL': methods.QEI_CL,
        'LP_EI': methods.LP_EI,
        'BLCB': methods.BLCB,
        'Random_EI': methods.Random_EI,
        'Random': methods.Random
    }

    kernels_gpflow = {
        'RBF': gpflow.kernels.RBF,
        'Matern32': gpflow.kernels.Matern32,
        'Matern52': gpflow.kernels.Matern52
    }

    kernels_gpy = {
        'RBF': GPy.kern.RBF,
        'Matern32': GPy.kern.Matern32,
        'Matern52': GPy.kern.Matern52
    }

    options = vars(copy.copy(args))
    options['objective'] = functions[options['function']]
    options['objective'].bounds = np.asarray(options['objective'].bounds)
    options['objective'] = scale_function(options['objective'])

    input_dim = options['objective'].bounds.shape[0]
    if options['algorithm'] != 'LP_EI':
        options['kernel'] = kernels_gpflow[options['kernel']](
            input_dim=input_dim, ARD=options['ard']
        )
        if options['priors']:
            options['kernel'].lengthscales.prior = gpflow.priors.Gamma(shape=2, scale=0.5)
            options['kernel'].variance.prior = gpflow.priors.Gaussian(mu=1, var=1)
        if options['samples'] > 0:
            assert options['priors']
    else:
        options['kernel'] = kernels_gpy[options['kernel']](
            input_dim=input_dim, ARD=options['ard']
        )

    options['job_name'] = options['function'] + '_' + options['algorithm']
    bo = algorithms[options['algorithm']](options)

    # Save command line arguments
    save_folder = 'out/' + bo.options['job_name'] + '/'
    filepath = save_folder + 'arguments.pkl'
    try:
        os.makedirs(save_folder)
    except OSError:
        pass
    try:
        os.remove(filepath)
    except OSError:
        pass
    with open(filepath, 'wb') as file:
        pickle.dump(args, file, pickle.HIGHEST_PROTOCOL)

    seed_start = options['seed']
    seed_end = seed_start + options['num_seeds']
    # print(options)
    for seed in range(seed_start, seed_end):
        run(bo, seed, robust=options['robust'], save=options['save'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--function', default='branin')
    parser.add_argument('--algorithm', default='OEI')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--num_seeds', type=int, default=5)
    parser.add_argument('--robust', type=int, default=1)
    parser.add_argument('--save', type=int, default=1)

    parser.add_argument('--samples', type=int, default=0)
    parser.add_argument('--priors', type=int, default=0)

    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--initial_size', type=int, default=10)
    parser.add_argument('--model_restarts', type=int, default=20)
    parser.add_argument('--opt_restarts', type=int, default=20)
    parser.add_argument('--normalize_Y', type=int, default=1)
    parser.add_argument('--noise', type=float)
    parser.add_argument('--kernel', default='RBF')
    parser.add_argument('--ard', type=int, default=1)
    parser.add_argument('--nl_solver',  default='scipy')
    parser.add_argument('--hessian', type=int, default=0)

    parser.add_argument('--beta_multiplier', type=float, default=.1)
    parser.add_argument('--delta', type=float, default=.1)
    parser.add_argument('--liar_choice', default='mean')
    args = parser.parse_args()

    main(args)
