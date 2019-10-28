from gpflow.gpr import GPR
import numpy as np
import tensorflow as tf
import random
import copy
from .solvers import solve
from .sdp import reset_warm_starting
import logging.config
import shutil
import os
import re
import yaml
import traceback

class BO(GPR):
    '''
    This is a simple (abstract) implementation of Bayesian Optimization.
    It extends gpflow's GPR.
    '''
    def __init__(self, options):
        self.bounds = options['objective'].bounds.copy()
        self.dim = self.bounds.shape[0]

        if 'mean_function' not in options:
            options['mean_function'] = None
        super(BO, self).__init__(X=np.zeros((0, self.dim)),
                                 Y=np.zeros((0, self.dim)),
                                 kern=options['kernel'],
                                 mean_function=options['mean_function']
                                 )
        self.options = options.copy()

        # Fix the noise, if its value is provided
        if self.options['noise'] is not None:
            self.likelihood.variance = self.options['noise']
            self.likelihood.variance.fixed = True

    def bayesian_optimization(self):
        '''
        This function implements the main loop of the Bayesian Optimization
        '''
        # Copy the objective. This is essential when testing draws from GPs
        objective = copy.copy(self.options['objective'])
        # Initialize the dataset at some random locations
        X0 = self.random_sample(self.bounds, self.options['initial_size'])
        # Evaluate the objective funtion there
        ret = objective.f(X0)
        # The objective might alter (e.g. discretize) X0.
        # In this case it returns two matrices:
        # the objective values (y0) and the altered locations (X0)
        if isinstance(ret, tuple):
            y0, X0 = ret
        else:
            y0 = ret

        # Set the data to the GP model
        # Careful, we might normalize the function evaluations
        # Provide only the first column of y0.
        # The others columns contain auxiliary data.
        self.X = X0
        self.Y = self.normalize(y0[:, 0:1])

        # X_all stores all the points where f was evaluated and
        # y_all the respective function values
        X_all = X0
        y_all = y0

        #------------------------------------------------------------------
        # Logging - Information about the function and initial evaluations.
        self.setup_logging()
        logger = logging.getLogger('evals')
        logger.info('----------------------------')
        logger.info('Bounds:\n' + str(self.bounds))
        if hasattr(objective, 'fmin'):
            logger.info('Minimum value:' + str(objective.fmin))
        logger.info('----------------------------')
        for i in range(len(X0)):
            logger.info(
                'X:' + str(X0[i, :]) + ' y: ' + str(y0[i, :])
            )
        #------------------------------------------------------------------

        for i in range(self.options['iterations']):
            # ML (or MAP if there are priors) for the GP model
            self.optimize_restarts(restarts=self.options['model_restarts'])

            #------------------------------------------------------------------
            # Logging - Print hyperparameters of the model
            logging.getLogger('').info('#Iteration:' + str(i + 1))
            # Remove non-printable characters (bold identifiers)
            ansi_escape = re.compile(r'\x1b[^m]*m')
            logging.getLogger('model').info(ansi_escape.sub('', str(self)))
            #------------------------------------------------------------------

            # Get the suggested points for optimising the acquisition function
            X_new = self.get_suggestion(self.options['batch_size'])
            # Evaluate the black-box function at the suggested points
            ret = objective.f(X_new)
            # The objective might alter (e.g. discretize) X_new.
            # In this case it returns two matrices:
            # the objective values (y_new) and the altered locations (X_new)
            if isinstance(ret, tuple):
                y_new, X_new = ret
            else:
                y_new = ret

            # Append the algorithm's choice X_new to X_all
            # Add the function evaluations f(X_new) to y_all
            X_all = np.concatenate((X_all, X_new))
            y_all = np.concatenate((y_all, y_new))

            # Update the GP model
            # Careful, the model might normalize the function evaluations
            # Provide only the first column of y_all.
            # The others columns contain auxiliary data.
            self.X = X_all
            self.Y = self.normalize(y_all[:, 0:1])

            #------------------------------------------------------------------
            # Logging - Print new evaluations
            for j in range(len(X_new)):
                logging.getLogger('evals').info(
                    'X:' + str(X_new[j, :]) + ' y: ' + str(y_new[j, :])
                )
            #------------------------------------------------------------------

        return X_all, y_all

    def get_suggestion(self, batch_size):
        '''
        This function runs a descent-based optimization on the acquisition
        function to get a batch of bath_size evaluation points
        Inputs: batch_size: Number of evalution points
        Output: X: [batch_size x self.dim] ndarray with the evaluation points
        '''
        X = None    # Will hold the final decision
        y = None    # Will hold the acquisition function value at X
        # Tile bounds to match batch size
        bounds_tiled = np.tile(self.bounds, (batch_size, 1))
        # Run local gradient-descent optimizer multiple times
        # to avoid getting stuck in a poor local optimum
        for j in range(self.options['opt_restarts']):
            try:
                # Resets the warm starting of the SDP (sdp.py) as we start from
                # a random point that is not necessarily close to the previous ones.
                reset_warm_starting()

                # Initial point of the optimization
                X_init = self.random_sample(self.bounds, batch_size)
                y_init = self.acquisition(X_init)[0]

                X0, y0, status = solve(X_init=X_init,
                                    bounds=bounds_tiled,
                                    hessian=self.options['hessian'],
                                    bo=self,
                                    solver=self.options['nl_solver'])

                # Update X if the current local minimum is
                # the best one found so far
                if X is None or y0 < y:
                    X, y = X0, y0


                #--------------------------------------------------------
                # Logging - Print statistics for the non-linear solver
                logging.getLogger('opt').info(
                    '##Opt_it:' + str(j + 1) + ' Val:' + '%.2e' % y0 +
                    ' Diff:' + '%.2e' % (y_init - y0) +
                    ' It:' + str(status.nit)
                )
                if not status.success:
                    logging.getLogger('opt').warning(
                        'NLP Warning:' + str(status.message)
                    )
                #--------------------------------------------------------
            except AssertionError:# KeyboardInterrupt:
                raise
            '''
            except Exception as e:
                #--------------------------------------------------------
                # Logging - Report failure of the nonlinear solver
                logging.getLogger('info').critical(
                    'Optimization #' + str(j + 1) +
                    'of the acquisition function failed!\n Error:' + str(e)
                )
                #--------------------------------------------------------
            '''

        # Assert that at least one optimization run succesfully
        assert X is not None

        return X

    def optimize_restarts(self, restarts=1, **kwargs):
        '''
        Wrapper of gpflow's self._objective to allow for multiple
        random restarts on the maximization of the likelihood 
        of the GP's hyperparameters.
        '''
        self.compile()
        if self._needs_recompile:
            self.compile()
        obj = self._objective

        par_min = self.get_free_state().copy()
        val_min = obj(par_min)[0]
        for i in range(restarts):
            try:
                self.randomize()
                self.optimize(**kwargs)
                x = self.get_free_state().copy()
                val = obj(x)[0]
            except KeyboardInterrupt:
                raise
            except:
                print("Hyperparameter optimization #" + str(i) + " failed!")# + traceback.print_exception(e))
                traceback.print_exc()
                val = float("inf")

            if val < val_min:
                par_min = x
                val_min = val

        self.set_state(par_min)

    @staticmethod
    def random_sample(bounds, k):
        '''
        Generate a set of k n-dimensional points sampled uniformly at random
        Inputs:
            bounds: n x 2 dimenional array containing upper/lower bounds
                    for each dimension
            k: number of points
        Output: k x n array containing the sampled points
        '''
        # k: Number of points
        n = bounds.shape[0]  # Dimensionality of each point
        X = np.zeros((k, n))
        for i in range(n):
            X[:, i] = np.random.uniform(bounds[i, 0], bounds[i, 1], k)

        return X

    def normalize(self, Y):
        '''
        When normalization is enabled, this function normalizes the first
        collumn of Y to have zero mean and std one.

        Recall that the first collumn contains the output of the function under
        minimization. The rest of the collumns (if any) contain auxiliary data
        that are only used for inspection purposes.
        '''

        Y_ = Y.copy()
        if self.options['normalize_Y'] and np.std(Y[:, 0]) > 0:
            Y_[:, 0] = (Y[:, 0] - np.mean(Y[:, 0]))/np.std(Y[:, 0])

        return Y_

    def setup_logging(self):
        '''
        This function initialises the logging mechanism
        '''
        if 'seed' in self.options:
            self.log_folder = 'log/' + self.options['job_name'] + '/' + str(self.options['seed']) + '/'
        else:
            self.log_folder = 'log/' + self.options['job_name'] + '/'

        try:
            os.makedirs(self.log_folder)
        except OSError:
            pass
        # Load config file
        with open('logging.yaml', 'r') as f:
            config = f.read()
        # Prepend logging folder
        config = config.replace('PATH/', self.log_folder)
        # Setup logging
        logging.config.dictConfig(yaml.load(config))
        #------------------------------------------------------------------
