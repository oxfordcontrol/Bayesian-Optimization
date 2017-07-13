import GPy
import numpy as np


class BO():
    '''
    This is a simple (abstract) implementation of Bayesian Optimization.
    One should implement the function acq_fun_optimizer to instantiate objects
    '''
    def __init__(self, options):
        '''
        The constructor just unpacks the various parameters given in the
        options dictionary
        '''

        # Remember to copy self.objective before using it
        # This is essential when testing draws from GPs
        self.objective = options['objective']
        self.bounds = self.objective.bounds
        self.dim = self.bounds.shape[0]

        self.kernel = options['kernel'].copy()

        self.noiseless = options['noiseless']
        if 'noise' in options:
            self.noise = options['noise']

        self.iterations = options['iterations']
        self.batch_size = options['batch_size']
        self.initial_size = options['initial_size']
        self.normalize_Y = options['normalize_Y']

        self.gp_opt_restarts = options['gp_opt_restarts']
        self.acq_opt_restarts = options['acq_opt_restarts']
        if 'optimizer_options' in options:
            self.optimizer_options = options['optimizer_options']
        else:
            self.optimizer_options = {}

        self.options = options.copy()

    def bayesian_optimization(self, X0, y0, objective):
        '''
        This function implements the main loop of Bayesian Optimization,
        starting by an initial set of evaluations y0 = objective.f(X0).
        It returns the final set of points X_all, the respective function
        outputs y_all, and a list with GP models at each iteration of the
        BO loop.

        The main work is done in the following three lines:

        ---------------------------------
        X_new = self.acq_fun_optimizer(m)
        ---------------------------------
        The new batch of points X_new is chosen
        by optimizing the acquisition function

        ---------------------------------------------------
        y_all = np.concatenate((y_all, objective.f(X_new)))
        ---------------------------------------------------
        The actual objective function (objective.f) is called on the
        selected points and the output is saved.

        ------------------------------------------------------
        m.optimize_restarts(num_restarts=self.gp_opt_restarts,
                            verbose=False, robust=True)
        ------------------------------------------------------
        The GP is updated & retrained to also take into account the newly
        acquired points
        '''

        # Set up GP model
        # Careful, the model might normalize the function evaluations
        # Provide only the first column of y0.
        # The others columns contain auxiliary data.
        m = GPy.models.GPRegression(X0, self.normalize(y0[:, 0:1]),
                                    self.kernel.copy())
        if self.noiseless:
            m.Gaussian_noise.constrain_fixed(1e-6, warning=False)
        elif hasattr(self, 'noise'):
            m.Gaussian_noise.constrain_fixed(self.noise, warning=False)

        # Train the GP model
        if self.gp_opt_restarts > 0:
            m.optimize_restarts(num_restarts=self.gp_opt_restarts,
                                verbose=False, robust=True)

        # List for saving GP models at every iteration (used for inspection)
        models = []
        # Save initial model
        models.append(m.param_array.copy())

        # X_all stores all the points where f was evaluated and
        # y_all the respective function values
        X_all = X0
        y_all = y0
        for i in range(self.iterations):
            X_new = self.acq_fun_optimizer(m)
            # Append the algorithm's choice X_new to X_all
            # Add the function evaluations f(X_new) to y_all
            X_all = np.concatenate((X_all, X_new))
            y_all = np.concatenate((y_all, objective.f(X_new)))

            # Update the GP model
            # Careful, the model might normalize the function evaluations
            # Provide only the first column of y_all.
            # The others columns contain auxiliary data.
            m.set_XY(X_all, self.normalize(y_all[:, 0:1]))

            # Retrain GP model
            if self.gp_opt_restarts > 0:
                m.optimize_restarts(num_restarts=self.gp_opt_restarts,
                                    verbose=False, robust=True)

            # Save model
            models.append(m.param_array.copy())

        return X_all, y_all, models

    def acq_fun_optimizer(self, m):
        raise NotImplementedError

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
        if self.normalize_Y and np.std(Y[:, 0]) > 0:
            Y_[:, 0] = (Y[:, 0] - np.mean(Y[:, 0]))/np.std(Y[:, 0])

        return Y_

    def derivative_check(self, acq_fun, batch_size):
        '''
        Compares the derivative of an acquisition function
        (given by acq_fun(x)[1]) against numerical differentiation
        '''

        try:
            import numdifftools as nd
        except ImportError:
            print('IMPORT ERROR:',
                  'Please install numdifftools to test derivatives.')
            raise

        # Number of test locations
        N = 100

        errors = np.zeros(N)

        for i in range(N):
            x_test = self.random_sample(self.bounds, batch_size).flatten()

            # Calculate derivative via numdifftools library
            def f(x): return acq_fun(x)[0]
            derivative_num = nd.Gradient(f)(x_test)

            derivative = acq_fun(x_test)[1]

            errors[i] = np.linalg.norm(derivative - derivative_num) /\
                np.linalg.norm(derivative)

            print('Value:', acq_fun(x_test)[0])
            print('Analytical derivative:', derivative)
            print('Numerical derivative:', derivative_num)
            print('Normalized Error:', errors[i]*100, '%')
            if errors[i] > 0.01:
                print('WARNING: found a point with considerable difference',
                      'between the analytical and the numerical derivative')
                print('Location:', x_test)
                input('Press Enter to continue...')

        print('Mean normalized error:', np.mean(errors))
