from __future__ import print_function
import gpflow
import tensorflow as tf
import numpy as np
from gpflow.param import Param
from gpflow import transforms
from gpflow._settings import settings

float_type = settings.dtypes.float_type
int_type = settings.dtypes.int_type
np_float_type = np.float32 if float_type is tf.float32 else np.float64


class SafeMatern52(gpflow.kernels.Matern52):
    # https://github.com/GPflow/GPflow/issues/421
    def euclid_dist(self, X, X2):
        r2 = self.square_dist(X, X2)
        return tf.sqrt(r2 + 1e-9)  # or 1e-6. I find 1e-12 is too small


class SafeMatern32(gpflow.kernels.Matern32):
    # https://github.com/GPflow/GPflow/issues/421
    def euclid_dist(self, X, X2):
        r2 = self.square_dist(X, X2)
        return tf.sqrt(r2 + 1e-9)  # or 1e-6. I find 1e-12 is too small


class NN(gpflow.kernels.Kern):
    """
    NN kernel
    """ 

    def __init__(self, input_dim,
                 x0=None,
                 variance=1.0, weight_variances=1., bias_variance=1.,
                 active_dims=None, ARD=False):
        """
        - input_dim is the dimension of the input to the kernel
        - order specifies the activation function of the neural network
          the function is a rectified monomial of the chosen order.
        - variance is the initial value for the variance parameter
        - weight_variances is the initial value for the weight_variances parameter
          defaults to 1.0 (ARD=False) or np.ones(input_dim) (ARD=True).
        - bias_variance is the initial value for the bias_variance parameter
          defaults to 1.0.
        - active_dims is a list of length input_dim which controls which
          columns of X are used.
        - ARD specifies whether the kernel has one weight_variance per dimension
          (ARD=True) or a single weight_variance (ARD=False).
        """
        gpflow.kernels.Kern.__init__(self, input_dim, active_dims)

        self.variance = Param(variance, transforms.positive)
        self.bias_variance = Param(bias_variance, transforms.positive)
        if ARD:
            if weight_variances is None:
                weight_variances = np.ones(input_dim, np_float_type)
            else:
                # accepts float or array:
                weight_variances = weight_variances * np.ones(input_dim, np_float_type)
            self.weight_variances = Param(weight_variances, transforms.positive)
            self.ARD = True
        else:
            if weight_variances is None:
                weight_variances = 1.0
            self.weight_variances = Param(weight_variances, transforms.positive)
            self.ARD = False

        if x0 is None:
            x0 = np.zeros(input_dim, np_float_type)
        self.x0 = Param(x0, transforms.Logistic(-0.5, 0.5))

    def _weighted_product(self, X, X2=None):
        if X2 is None:
            return tf.reduce_sum(self.weight_variances * tf.square(X), axis=1) + self.bias_variance
        else:
            return tf.matmul((self.weight_variances * X), X2, transpose_b=True) + self.bias_variance

    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self._slice(X, X2)

        X_ = tf.subtract(X, self.x0[None, :])

        X_denominator = tf.sqrt(1 + 2*self._weighted_product(X_))
        if X2 is None:
            X2_ = X_
            X2_denominator = X_denominator
        else:
            X2_ = tf.subtract(X2, self.x0[None, :])
            X2_denominator = tf.sqrt(1 + 2*self._weighted_product(X2_))

        numerator = 2*self._weighted_product(X_, X2_)
        fraction = numerator / X_denominator[:, None] / X2_denominator[None, :]

        return self.variance * (2. / np.pi) * tf.asin(fraction)

    def Kdiag(self, X, presliced=False):
        # TODO: Make only the necessary calculations
        return tf.diag_part(self.K(X))
