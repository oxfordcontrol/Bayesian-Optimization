from .bo import BO
import numpy as np
import scipy as sp


class BLCB(BO):
    '''
    This class is an implementation of the Batch Upper Confidence Bound
    acquisition function. See the original MATLAB implementation.

    See:
    Desautels, Krause, and Burdick, JMLR, 2014
    Parallelizing Exploration-Exploitation Tradeoffs in
    Gaussian Process Bandit Optimization
    '''
    def __init__(self, options):
        # The options dictionary should include the entries
        # beta_multiplier and delta
        super(BLCB, self).__init__(options)
        self.beta_multiplier = options['beta_multiplier']
        self.delta = options['delta']

    @classmethod
    def acquisition_fun(cls, X, m, beta):
        ############################################################
        # Copied from GPyOpt/models/gpmodel/predict_withGradients
        # (changed to predict_noiseless)
        ############################################################
        mu, v = m.predict_noiseless(X)
        v = np.clip(v, 1e-10, np.inf)
        s = np.sqrt(v)

        dmudx, dvdx = m.predictive_gradients(X)
        dmudx = dmudx[:, :, 0]
        dsdx = dvdx / (2*np.sqrt(v))
        ###############

        opt_val = mu - beta * s
        dpdX = dmudx - beta * dsdx

        return opt_val, dpdX

    @classmethod
    def acquisition_fun_flat(cls, X, m, beta):
        '''
        Wrapper for acquisition_fun, where X is considered as a vector
        '''
        n = m.X.shape[1]
        k = X.shape[0]//n

        (opt_val, dpdX) = cls.acquisition_fun(X.reshape(k, n), m, beta)
        return opt_val, dpdX.flatten()

    def acq_fun_optimizer(self, m):
        # X_final will hold the final choice for the whole batch
        X_final = np.zeros((0, self.dim))
        for i in range(self.batch_size):
            # X will hold the final choice of the i_th point in the batch
            X = None
            # Y will hold the acquisition function value of X
            y = None

            # Calculate beta based on the beta_multiplier and delta choices.
            # Default values from Thomas Desautels' Code
            # delta = .1;
            # beta_multiplier = .1;
            beta = 2 * self.beta_multiplier * \
                np.log(self.dim * np.square(np.pi * (i + 1)) / 6 / self.delta)
            # Run local gradient-descent optimizer multiple times
            # to avoid getting stuck in a poor local optimum
            for j in range(self.acq_opt_restarts):
                X0 = self.random_sample(self.bounds, 1)
                res = sp.optimize.minimize(fun=self.acquisition_fun_flat,
                                           x0=X0.flatten(),
                                           args=(m, beta),
                                           method='L-BFGS-B',
                                           jac=True,
                                           bounds=self.bounds,
                                           options=self.optimizer_options)
                '''
                self.derivative_check(
                    lambda X: self.acquisition_fun_flat(X, m, beta), 1
                    )
                '''
                X0 = res.x.reshape(1, self.dim)
                y0 = res.fun[0]

                # Update X if the current local minimum is
                # the best one found so far
                if X is None or y0 < y:
                    X = X0
                    y = y0

            # Append i_th choice to the batch
            X_final = np.concatenate((X_final, X))

        return X_final
