from __future__ import print_function
from .bo import BO
import numpy as np
from gpflow.param import AutoFlow
from gpflow._settings import settings
import tensorflow as tf
float_type = settings.dtypes.float_type

BATCH_SIZE = -1


class QEI_CL(BO):
    '''
    This class is an implementation of the constant liar heuristic for using
    Expected Improvement in the batch case.

    See:
    Ginsbourger D., Le Riche R., Carraro L. (2010)
    Kriging Is Well-Suited to Parallelize Optimization.
    '''
    def __init__(self, options):
        global BATCH_SIZE
        BATCH_SIZE = options['batch_size']
        super(QEI_CL, self).__init__(options)
        # The options dictionary should include the entry 'liar_choice'
        # Its value should be 'min', 'max' or 'mean'
        self.liar_choice = options['liar_choice']

    @AutoFlow((float_type, [None, None]))
    def acquisition_tf(self, X):
        fmin = tf.reduce_min(self.build_predict(self.X)[0])
        mean, var = self.likelihood.predict_mean_and_var(*self.build_predict(X)) 
        normal = tf.contrib.distributions.Normal(mean, tf.sqrt(var))
        f = -(fmin - mean) * normal.cdf(fmin) - var * normal.prob(fmin)
        df = tf.gradients(f, X)[0]
        return tf.reshape(f, [-1]), tf.reshape(df, [-1])

    @AutoFlow((float_type, [None]))
    def acquisition_hessian(self, x):
        X = tf.reshape(x, [BATCH_SIZE, -1])
        fmin = tf.reduce_min(self.build_predict(self.X)[0])
        mean, var = self.likelihood.predict_mean_and_var(*self.build_predict(X)) 
        normal = tf.contrib.distributions.Normal(mean, tf.sqrt(var))
        f = -(fmin - mean) * normal.cdf(fmin) - var * normal.prob(fmin)
        return tf.hessians(f, x)[0]

    def get_suggestion(self, batch_size):
        if self.liar_choice == 'min':
            y_liar = np.min(self.Y.value).reshape(1, 1)
        elif self.liar_choice == 'max':
            y_liar = np.max(self.Y.value).reshape(1, 1)
        elif self.liar_choice == 'mean':
            y_liar = np.mean(self.Y.value).reshape(1, 1)

        # Copy original observations
        X_orig = self.X.value.copy()
        Y_orig = self.Y.value.copy()

        # X_final will hold the final choice for the whole batch
        X_final = np.zeros((0, self.dim))
        for i in range(self.batch_size):

            X = super(QEI_CL, self).get_suggestion(batch_size=1)

            # Append the liar in the model
            # Don't normalize the function evaluations, since the liar values
            # are calculaled based on self.Y, which is normalized
            # Also, normalizing again would require to retrain the model, but
            # we don't want to do that based on liar values.
            self.X = np.concatenate((self.X.value, X))
            self.Y = np.concatenate((self.Y.value, y_liar))

            # Append i_th choice to the batch
            X_final = np.concatenate((X_final, X))

        # Revert model to the original observations
        self.X = X_orig
        self.Y = Y_orig

        return X_final
