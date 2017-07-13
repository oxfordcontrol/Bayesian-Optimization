from multiprocessing import Pool
from .bo import BO
import GPy
import git
import numpy as np
import matplotlib.pyplot as plt
import copy
import pickle


class Caller():
    '''
    A class that, given a BO object, runs multiple Bayesian Optimization loops
    and keeps the results for analysis. It includes methods for plotting,
    saving/loading data.
    '''
    def __init__(self, job_name=None, bo=None, filename=None):
        '''
        The user can provide either an object of type BO or
        the options dictionary used to create the BO object.
        The first case is used to run and/or plot experiments.
        The second case is only used to plot saved experiments.
        The user cannot provide both, as the BO object has its own options.
        '''
        assert (job_name is None and bo is None) or filename is None, \
            'You cannot provide both a BO object and a filename to load from'
        if filename is None:
            self.bo = bo

            self.job_name = job_name

            self.seeds = []   # List with the seeds that led to successful runs
            self.failed_seeds = []  # Same, but for failed runs
            self.points = []  # List of the resulting points of evaluation for
            # each successful run (matching with the self.seeds list)
            self.outputs = []   # Same, but for the function outputs
            self.models = []    # List of lists with the GP model for each
            # iteration of the BO loop for every successful run
        else:
            self.load_data(filename)

    def run(self, seed, X0=None, y0=None, robust=True, save=False):
        '''
        Runs a single Bayesian Optimization loop. Basically the function
        is a fancy wrapper to the following call:
            self.bo.bayesian_optimization(X0, y0, objective)

        Inputs:
            seed: Random seed for the Experiment
            X0: Array with the locations of the initial evaluations
            y0: f(X0)
            robust: If true, the function catches any error produced and
                    returns gracefully.
            save: If true, append the BO loop results to the object's lists.
        Outputs:
            points, outputs: Final set of evaluation locations and
                             outputs of the BO loop
            models: List of the GP models at each iteration of the BO loop
        '''
        points = None
        outputs = None
        models = None

        # Copy the objective. This is essential when testing draws from GPs
        objective = copy.copy(self.bo.objective)

        np.random.seed(seed)

        if X0 is None:
            X0 = BO.random_sample(self.bo.bounds, self.bo.initial_size)
            y0 = objective.f(X0)
        else:
            assert y0 is not None
            assert X0.shape[0] == y0.shape[0] == self.bo.initial_size

        failed = False
        if robust:
            try:
                points, outputs, models = \
                    self.bo.bayesian_optimization(X0, y0, objective)
                print('Done with seed:', seed)
            except:
                failed = True
                print('Experiment of', self.job_name,
                      'with seed', seed, 'failed')
        else:
            points, outputs, models = \
                self.bo.bayesian_optimization(X0, y0, objective)
            print('Done with seed:', seed)

        if save:
            if failed:
                self.append_data(failed_seeds=[seed])
            else:
                self.append_data(points=[points],
                                 outputs=[outputs],
                                 models=[models],
                                 seeds=[seed])

        return points, outputs, models

    def run_multiple(self, seeds, num_threads=4):
        '''
        Runs multiple BO loops in parallel and appends the results to the
        object's lists.
        '''
        pool = Pool(num_threads)
        results = pool.map(self.run, seeds)
        pool.close()
        pool.join()

        succeeded_seeds = []
        failed_seeds = []

        points = []
        outputs = []
        models = []
        for i in range(len(results)):
            if results[i][0] is None:
                failed_seeds.append(seeds[i])
            else:
                succeeded_seeds.append(seeds[i])

                points.append(results[i][0])
                outputs.append(results[i][1])
                models.append(results[i][2])

        self.append_data(points=points, outputs=outputs, models=models,
                         seeds=succeeded_seeds, failed_seeds=failed_seeds)

    def plot(self, color='b', fig=None, ax=None, label=None, output_idx=0,
             offset=0):
        '''
        Plot error bars for the minimum achieved at each iteration of the BO
        Inputs:
            color: string specifying the color (according to matplotlib)
            fig: Matplotlib figure object to draw to. If None create a new one.
            ax: Matplotlib ax object to draw to. If None create a new one.
            label: String that will appear in the legend
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
        if label is None:
            repo = git.Repo(search_parent_directories=True)
            sha = repo.head.object.hexsha[0:7]
            label = self.job_name + '_' + sha

        n = len(self.seeds)  # Number of successful experiments
        k = self.bo.iterations  # Number of iterations
        mins = np.zeros((n, k + 1))
        for i in range(n):
            for j in range(k + 1):
                idx = np.argmin(self.outputs[i][0:self.bo.initial_size +
                                                j*self.bo.batch_size, 0])
                mins[i, j] = self.outputs[i][idx, output_idx]

        fig, ax = self.plot_mins(mins, color, fig, ax, label, offset)

        return fig, ax

    def plot_random(self, color='b', fig=None, ax=None, label=None,
                    output_idx=0, offset=0):
        '''
        Same with plot(), but plot the results of a random uniform strategy
        '''
        if label is None:
            label = 'random uniform'

        all_seeds = self.seeds + self.failed_seeds
        n = len(all_seeds)  # Num of ALL experiments
        k = self.bo.iterations  # Number of iterations
        mins = np.zeros((n, k + 1))
        for i in range(n):
            objective = copy.copy(self.bo.objective)
            np.random.seed(int(all_seeds[i]))
            # We generate X in two steps so as the first initial_size points
            # are the same as the ones in the real experiments, as dictated by
            # the seed. This is important when testing draws of a Gaussian
            # Process.
            X0 = BO.random_sample(self.bo.bounds, self.bo.initial_size)
            Y0 = objective.f(X0)

            X = BO.random_sample(self.bo.bounds, k*self.bo.batch_size)
            Y = objective.f(X)

            X = np.concatenate((X0, X))
            Y = np.concatenate((Y0, Y))
            for j in range(k + 1):
                idx = np.argmin(Y[0:self.bo.initial_size +
                                j*self.bo.batch_size, 0])
                mins[i, j] = Y[idx, output_idx]

        fig, ax = self.plot_mins(mins, color, fig, ax, label, offset)

        return fig, ax

    def plot_mins(self, mins, color='b', fig=None, ax=None, label=None,
                  offset=0):
        # Auxiliary function used by plot(), plot_random()
        if fig is None or ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        ax.errorbar(np.arange(0, self.bo.iterations + 1) + offset,
                    np.mean(mins, axis=0), yerr=np.std(mins, axis=0),
                    capsize=0, color=color, label=label, fmt='.')
        ax.legend()

        return fig, ax

    def print_models(self, lengthscales=True, variance=True, noise=True):
        # Prints the parameters of all GP models
        for i in range(len(self.seeds)):
            print('Seed:', self.seeds[i])
            for j in range(len(self.models[i])):
                print('Iteration:', j)

                m = self.get_model(i, j)
                print('Variance:', m.kern.variance)
                print('Lengthscale:', m.kern.lengthscale)
                print('Noise var:', m.likelihood.variance)
                input('Press Enter to continue...')

    def get_model(self, seed_idx, iteration):
        '''
        Recreates the GP model of the given starting seed and
        iteration of the BO loop.
        '''
        idx = self.bo.initial_size + iteration*self.bo.batch_size
        X = self.points[seed_idx][0:idx]
        Y = self.outputs[seed_idx][0:idx]
        m_params = self.models[seed_idx][iteration]

        # Copied from https://github.com/SheffieldML/GPy/README.md
        # Provide only the first column of Y.
        # The others columns contain auxiliary data.
        m_load = GPy.models.GPRegression(X, self.bo.normalize(Y[:, 0:1]),
                                         self.bo.kernel.copy(),
                                         initialize=False)
        # do not call the underlying expensive algebra on load
        m_load.update_model(False)
        # initialize the parameters (connect the parameters up)
        m_load.initialize_parameter()
        m_load[:] = m_params  # Load the parameters
        m_load.update_model(True)  # Call the algebra only once

        return m_load

    def append_data(self, points=[], outputs=[], models=[], seeds=[],
                    failed_seeds=[]):
        # Appends results to the lists of the object
        self.seeds = self.seeds + seeds
        self.failed_seeds = self.failed_seeds + failed_seeds

        self.points = self.points + points
        self.outputs = self.outputs + outputs
        self.models = self.models + models

    def load_data(self, filename=None):
        '''
        Loads data from a file
        Warning: pickle is used, so the saving/loading might be inconsistent
        across different versions
        '''
        if filename is None:
            repo = git.Repo(search_parent_directories=True)
            sha = repo.head.object.hexsha[0:7]
            filename = self.job_name + '_' + sha
            filename = 'results/' + filename + '.dat'
        with open(filename, 'rb') as f:
            (options, self.job_name, self.seeds, self.failed_seeds,
                self.points, self.outputs, self.models) = pickle.load(f)
            self.bo = BO(options)

    def save_data(self, filename=None):
        '''
        Saves data to a file
        Warning: pickle is used, so the saving/loading might be inconsistent
        across different versions
        '''
        data = (self.bo.options, self.job_name,
                self.seeds, self.failed_seeds,
                self.points, self.outputs, self.models)

        if filename is None:
            repo = git.Repo(search_parent_directories=True)
            sha = repo.head.object.hexsha[0:7]
            filename = self.job_name + '_' + sha
            filename = 'results/' + filename + '.dat'

        with open(filename, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
