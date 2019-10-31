import methods.config
import os
import numpy as np
import tensorflow as tf
import random
import argparse
import gpflow
from methods.oei import OEI
from methods.random import Random
import time
import pickle
from benchmark_functions import scale_function, hart6, eggholder
import copy

algorithms = {
    'OEI': OEI,
    'Random': Random
}

class SafeMatern32(gpflow.kernels.Matern32):
    # See https://github.com/GPflow/GPflow/pull/727
    def euclid_dist(self, X, X2):
        r2 = self.square_dist(X, X2)
        return tf.sqrt(tf.maximum(r2, 1e-40))


def run(options, seed, robust=False, save=False):
    '''
    Runs bayesian optimization on the setup defined in the options dictionary
    starting from a predefined seed. Saves results on the folder named 'out' while logging 
    is saved on the folder 'log'.
    '''
    options['seed'] = seed
    # Set random seed: Numpy, Tensorflow, Python 
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Create bo object which will be called later to perform Bayesian Optimization.
    bo = algorithms[options['algorithm']](options)

    try:
        start = time.time()
        # Run BO
        X, Y = bo.bayesian_optimization()
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


def create_options(args):
    functions = {
        'hart6': hart6(),
        'eggholder': eggholder()
    }

    kernels_gpflow = {
        'RBF': gpflow.kernels.RBF,
        'Matern32': SafeMatern32,
    }

    options = vars(copy.copy(args))
    options['objective'] = functions[options['function']]
    options['objective'].bounds = np.asarray(options['objective'].bounds)
    # This scales the input domain of the function to [-0.5, 0.5]^n. It's different to the
    # normalize option, which scales the output of the function.
    options['objective'] = scale_function(options['objective'])

    input_dim = options['objective'].bounds.shape[0]
    if options['algorithm'] != 'LP_EI':
        k = kernels_gpflow[options['kernel']](
            input_dim=input_dim, ARD=options['ard'])
        if options['priors']:
            k.lengthscales.prior = gpflow.priors.Gamma(shape=2, scale=0.5)
            k.variance.prior = gpflow.priors.Gaussian(mu=1, var=2)
        options['kernel'] = k

    options['job_name'] = options['function'] + '_' + options['algorithm']

    return options


def main(args):
    options = create_options(args)

    save_folder = 'out/' + options['job_name'] + '/'
    filepath = save_folder + 'arguments.pkl'
    try:
        os.makedirs(save_folder)
    except OSError:
        pass

    try:
        os.remove(filepath)
    except OSError:
        pass
    try:
        with open(filepath, 'wb') as file:
            pickle.dump(args, file, pickle.HIGHEST_PROTOCOL)
    except OSError:
        pass

    filepath = save_folder + 'fmin.txt'
    try:
        fmin = options['objective'].fmin
    except AttributeError:
        fmin = 0
    np.savetxt(filepath, np.array([fmin]))

    for seed in range(args.seed, args.seed + args.num_seeds):
        run(options, seed=seed, save=options['save'])


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--function', default='hart6')
    parser.add_argument('--algorithm', default='OEI')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--num_seeds', type=int, default=1)
    parser.add_argument('--save', type=int, default=1)

    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--initial_size', type=int, default=10)
    parser.add_argument('--model_restarts', type=int, default=20,
        help='Random restarts when optimizing the Likelihood of the GP.')
    parser.add_argument('--opt_restarts', type=int, default=20,
        help='Random restarts when optimizing the acquisition function.')
    parser.add_argument('--normalize_Y', type=int, default=1,
        help='If set to 1, then the outputs of the function under optimization is normalized to have variance 1 and mean 0')
    parser.add_argument('--noise', type=float,
        help='Used to set the likelihood to a fixed value')
    parser.add_argument('--kernel', default='Matern32')
    parser.add_argument('--ard', type=int, default=0)
    parser.add_argument('--nl_solver',  default='knitro')
    parser.add_argument('--hessian', type=int, default=1)

    parser.add_argument('--priors', type=int, default=0)

    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    main(args)

