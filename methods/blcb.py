from __future__ import print_function
from .bo import BO
import numpy as np
from gpflow.param import AutoFlow
from gpflow._settings import settings
import tensorflow as tf
float_type = settings.dtypes.float_type


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

    def acquisition(self, x):
        k = x.size // self.dim
        X = x.reshape(k, self.dim)

        obj, gradient = self.acquisition_tf(X, self.beta)

        return obj, gradient

    @AutoFlow((float_type, [None, None]), (float_type, []))
    def acquisition_tf(self, X, beta):
        mean, var = self.likelihood.predict_mean_and_var(*self.build_predict(X))
        f = mean - beta*tf.sqrt(var)
        df = tf.gradients(f, X)[0]
        return tf.reshape(f, [-1]), tf.reshape(df, [-1])

    def get_suggestion(self, batch_size):
        # X_final will hold the final choice for the whole batch
        X_final = np.zeros((0, self.dim))
        for i in range(batch_size):
            # Calculate beta based on the beta_multiplier and delta choices.
            # Default values from Thomas Desautels' Code
            # delta = .1;
            # beta_multiplier = .1;
            self.beta = 2 * self.beta_multiplier * \
                np.log(self.dim * np.square(np.pi * (i + 1)) / 6 / self.delta)

            X = BO.get_suggestion(self, batch_size=1)

            # Append i_th choice to the batch
            X_final = np.concatenate((X_final, X))

        return X_final
