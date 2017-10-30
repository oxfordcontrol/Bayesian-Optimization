import numpy as np
import argparse
import gpflow
import GPy
import tensorflow as tf
import random
import methods
from test_functions import gp
import copy
import os
import matplotlib.pyplot as plt
# When no X server is present
plt.switch_backend('agg')


class Quadratic(gpflow.mean_functions.MeanFunction):
    """
    Quadratic mean function for gpflow (see gpflow.mean_functions)
    y_i = x_i^T (A A^T) x_i
    """
    def __init__(self, A):
        """
        If X has N rows and D columns, and Y is intended to have 1 column,
        then A must be D x K, where 1 <= K <= D
        """
        gpflow.mean_functions.MeanFunction.__init__(self)
        self.A = gpflow.param.Param(np.atleast_2d(A))

    def __call__(self, X):
        return tf.reduce_sum(tf.square(tf.matmul(X, self.A)),
                             axis=1, keep_dims=True)


def qei_sampling(mean, cov, fmin, N=100000):
    imp = np.zeros(N)
    for k in range(N):
        Y = np.random.multivariate_normal(mean[:, 0], cov[:, :, 0])[:, None]
        imp[k] = min(fmin, min(Y)) - fmin

    return np.mean(imp)


def main(args):
    algorithms = {
        'OEI': methods.OEI,
        'QEI': methods.QEI,
        'QEI_CL': methods.QEI_CL,
        'LP_EI': methods.LP_EI,
        'BLCB': methods.BLCB,
        'Random_EI': methods.Random_EI,
        'Random': methods.Random
    }

    options = vars(copy.copy(args))
    variance = 10
    lengthscales = np.asarray([1/10])
    input_dim = lengthscales.size
    bounds = np.repeat(np.array([[-1, 1]]), input_dim, axis=0)

    options['kernel'] = gpflow.kernels.RBF(
        input_dim=input_dim, ARD=options['ard'],
        variance=variance, lengthscales=lengthscales,
    )

    A = np.array([[5]])
    mean_function = Quadratic(A=np.array([[5]]))
    options['mean_function'] = mean_function
    options['objective'] = gp(kernel=options['kernel'], bounds=bounds,
                              sd=np.sqrt(options['noise']),
                              mean_function=mean_function)

    options['job_name'] = 'gp_' + options['algorithm']

    options_gpy = options.copy()
    options_gpy['kernel'] = GPy.kern.RBF(
        input_dim=input_dim, ARD=options['ard'],
        variance=variance, lengthscale=lengthscales,
    )

    mean_function_gpy = GPy.core.Mapping(1, 1)
    mean_function_gpy.f = lambda x: np.sum(
        np.square(x.dot(A)), axis=1, keepdims=True
        )
    mean_function_gpy.update_gradients = lambda a, b: None
    options_gpy['mean_function'] = mean_function_gpy

    if not args.plot_posteriors:
        num_batches = args.batch_size_max - args.batch_size + 1
        results = np.zeros((num_batches, options['num_seeds']))
        results_sampling = np.zeros((num_batches, options['num_seeds']))
        for batch_size in range(args.batch_size, args.batch_size_max + 1):
            batch_idx = batch_size - args.batch_size
            options['batch_size'] = batch_size
            options_gpy['batch_size'] = batch_size

            seed_start = args.seed
            seed_end = seed_start + options['num_seeds']
            for seed in range(seed_start, seed_end):
                options['seed'] = seed
                tf.reset_default_graph()
                tf.set_random_seed(seed)
                np.random.seed(seed)
                random.seed(seed)

                seed_idx = seed - seed_start

                options['job_name'] = 'gp_' + options['algorithm']
                if options['algorithm'] != 'LP_EI':
                    bo = algorithms[options['algorithm']](options)
                else:
                    bo = algorithms[options_gpy['algorithm']](options_gpy)

                X, Y = bo.bayesian_optimization()

                X_init = X[:options['initial_size'], :]
                Y_init = Y[:options['initial_size'], 0:1]

                X_choice = X[-options['batch_size']:, :]

                # Create objects that will act as evaluators for qei, oei
                qei_evaluator = methods.QEI(options)
                oei_evaluator = methods.OEI(options)
                if options['noise'] is not None:
                    oei_evaluator.likelihood.variance = options['noise']
                    oei_evaluator.likelihood.variance.fixed = True

                    qei_evaluator.likelihood.variance = options['noise']
                    qei_evaluator.likelihood.variance.fixed = True

                qei_evaluator.X = X_init
                qei_evaluator.Y = Y_init
                oei_evaluator.X = X_init
                oei_evaluator.Y = Y_init

                qei_result = qei_evaluator.acquisition_tf(X_choice)[0]
                qei_sampling_result = qei_sampling(
                    *qei_evaluator.predict_f_full_cov(X_choice),
                    np.min(Y_init)
                )

                results[batch_idx][seed_idx] = qei_result
                results_sampling[batch_idx][seed_idx] = qei_sampling_result

                if args.plot_problem and \
                   np.absolute(qei_result - qei_sampling_result) > 1:
                    # We are in the case of an inaccurate calculation of qei
                    xx = np.linspace(-.5, .5, 101)
                    yy = np.zeros(xx.shape)
                    yy_o = np.zeros(xx.shape)
                    yy_s = np.zeros(xx.shape)
                    # Choose a random direction
                    X = np.random.rand(*X_choice.shape)
                    for i in range(len(xx)):
                        # Test nearby the problematic point
                        X_test = X_choice + xx[i]*X
                        # QEI
                        yy[i] = qei_evaluator.acquisition_tf(X_test)[0]
                        # OEI
                        yy_o[i] = oei_evaluator.acquisition(X_test)[0]
                        # QEI sampled
                        yy_s[i] = qei_sampling(
                            *qei_evaluator.predict_f_full_cov(X_test),
                            min(Y_init), N=10000
                        )

                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    ax.plot(xx, yy_s, color='k', linewidth=3, label='QEI (sampled)')
                    ax.plot(xx, yy, color='g', linewidth=2, label='QEI')
                    ax.plot(xx, yy_o, color='r', linewidth=2, label='OEI')
                    plt.yscale('symlog')
                    ax.legend()
                    ax.set_xlabel('t')
                    ax.set_ylabel('Expected Improvement')
                    fig.set_size_inches((4.5, 3))
                    fig.tight_layout()
                    plt.savefig('results/qei_problem_' + str(seed) + '.pdf')

        # Save results
        save_folder = 'out/' + bo.options['job_name'] + '/'
        filepath = save_folder + str(seed_start) + \
            '-' + str(seed_end - 1) + '.npz'
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        if os.path.isfile(filepath):
            os.remove(filepath)
        np.savez(filepath, results=results, results_sampling=results_sampling)
    else:
        # Just plot posteriors and the choices of all the below algorithms
        # Normally, for this section you would call the python script with
        # more opt_restarts, to ensure(?) global optimality.
        # Comment out the algorithms that you don't want in the plot
        algorithms_considered = {
            'OEI': methods.OEI,
            'QEI': methods.QEI,
            'QEI_CL': methods.QEI_CL,
            'LP_EI': methods.LP_EI,
            # 'BLCB': methods.BLCB,
            # 'Random_EI': methods.Random_EI,
            # 'Random': methods.Random
        }
        seed_start = args.seed
        seed_end = seed_start + options['num_seeds']
        for seed in range(seed_start, seed_end):
            options['seed'] = seed

            choices = []
            for name, algorithm in algorithms_considered.items():
                tf.reset_default_graph()
                tf.set_random_seed(seed)
                np.random.seed(seed)
                random.seed(seed)

                options['job_name'] = 'gp_' + name
                if name != 'LP_EI':
                    bo = algorithm(options)
                else:
                    bo = algorithm(options_gpy)
                X, Y = bo.bayesian_optimization()

                X_init = X[:options['initial_size'], :]
                Y_init = Y[:options['initial_size'], 0:1]
                X_choice = X[-options['batch_size']:, :]
                choices.append(X_choice)

            # Create auxiliary object to plot posterior
            qei_evaluator = methods.QEI(options)
            if options['noise'] is not None:
                qei_evaluator.likelihood.variance = options['noise']
                qei_evaluator.likelihood.variance.fixed = True

            qei_evaluator.X = X_init
            qei_evaluator.Y = Y_init
            fig = plt.figure()
            ax = fig.add_subplot(111)
            xx = np.linspace(-1, 1, 500)[:, None]
            mean, var = qei_evaluator.predict_y(xx)
            ax.plot(X_init, Y_init, 'ko', mew=2)
            ax.plot(xx, mean, 'b', lw=2)
            ax.fill_between(xx[:, 0], mean[:, 0] -
                            2*np.sqrt(var[:, 0]), mean[:, 0] +
                            2*np.sqrt(var[:, 0]), color='blue', alpha=0.2)
            ax.axis((-1, 1, -25, 30))
            ax.axis('off')
            i = 0
            colors = {'OEI': 'r', 'QEI': 'g', 'QEI_CL': 'b',
                      'LP_EI': 'y', 'BLCB': 'c', 'Random_EI': 'm'}
            plt.hold(True)
            for name, _ in algorithms_considered.items():
                x = choices[i][:, 0]
                print(name, ':', x)
                ax.plot(x, 0*x - i*3 - 10, linestyle='None',
                        marker='s', color=colors[name], ms=1.5,
                        label=name)
                i += 1

            ax.legend()
            plt.savefig('results/posterior_' + str(seed) + '.pdf')
            print(seed - seed_start, ":", int(end-start), end='|')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', default='OEI')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--num_seeds', type=int, default=1)
    parser.add_argument('--robust', type=int, default=1)

    parser.add_argument('--samples', type=int, default=0)
    parser.add_argument('--priors', type=int, default=0)

    parser.add_argument('--iterations', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--batch_size_max', type=int, default=5)
    parser.add_argument('--initial_size', type=int, default=10)
    parser.add_argument('--model_restarts', type=int, default=0)
    parser.add_argument('--opt_restarts', type=int, default=100)
    parser.add_argument('--normalize_Y', type=int, default=0)
    parser.add_argument('--noise', type=float, default=1e-6)
    parser.add_argument('--kernel', default='RBF')
    parser.add_argument('--ard', type=int, default=1)
    parser.add_argument('--nl_solver',  default='scipy')
    parser.add_argument('--hessian', type=int, default=0)

    parser.add_argument('--beta_multiplier', type=float, default=.1)
    parser.add_argument('--delta', type=float, default=.1)
    parser.add_argument('--liar_choice', default='mean')

    parser.add_argument('--plot_problem', type=int, default=0)
    parser.add_argument('--plot_posteriors', type=int, default=0)

    args = parser.parse_args()
    main(args)
