from .bo import BO
import numpy as np
import scipy.linalg as la
import logging
from .sdp import sdp, solution_derivative
import tensorflow as tf
from gpflow.param import AutoFlow
from gpflow._settings import settings
float_type = settings.dtypes.float_type


class OEI(BO):
    '''
    This class implements the Optimistic Expected Improvement acquisition function.
    '''
    def __init__(self, options):
        super(OEI, self).__init__(options)

    def acquisition(self, x):
        '''
        The acquisition function and its gradient
        Input: x: flattened ndarray [batch_size x self.dim] containing the evaluation points
        Output[0]: Value of the acquisition function
        Output[1]: Gradient of the acquisition function
        '''
        # Calculate the minimum value so far
        fmin = np.min(self.predict_f(self.X.value)[0])
        X = x.reshape((-1, self.dim))

        X_, V = self.project(X)
        # Solve the SDP
        value, M_, _, _ = sdp(self.omega(X_), fmin, warm_start=(len(X)==len(X_)))
        M = V.T.dot(M_).dot(V)
        _, gradient = self.acquisition_tf(X, M)
        return value, gradient

    def acquisition_hessian(self, x):
        '''
        The acquisition function and its gradient
        Input: x: flattened ndarray [batch_size x self.dim] containing the evaluation points
        Output[0]: Value of the acquisition function
        Output[1]: Gradient of the acquisition function
        '''
        # Calculate the minimum value so far
        fmin = np.min(self.predict_f(self.X.value)[0])
        X = x.reshape((-1, self.dim))

        # See comments on self.project
        X_, _ = self.project(X)
        if len(X) != len(X_):
            return np.zeros((x.shape[0], x.shape[0]))

        # Solve SDP
        _, M, Y, C = sdp(self.omega(X), fmin)
        # Calculate domega/dx and dM/dx
        domega = self.domega(X.flatten())
        dM = solution_derivative(M, Y, C, domega) 

        # Perform the chain rules in Tensorflow
        return self.acquisition_hessian_tf(X.flatten(), M, dM, domega)

    @AutoFlow((float_type, [None, None]), (float_type, [None, None]))
    def acquisition_tf(self, X, M):
        '''
        Calculates the acquisition function, given M the optimizer of the SDP.
        The calculation is simply a matrix inner product.
        Input: X: ndarray [batch_size x self.dim] containing the evaluation points
        Output[0]: Value of the acquisition function
        Output[1]: Gradient of the acquisition function
        '''
        f = tf.tensordot(self.omega_tf(X), M, axes=2)
        df = tf.gradients(f, X)[0]
        return tf.reshape(f, [-1]), tf.reshape(df, [-1])

    def omega_tf(self, X):
        '''
        Calculates the second order moment matrix in tensorflow.
        Input: X: ndarray [batch_size x self.dim] containing the evaluation points
        Output: [Sigma(X) + mu(X)*mu(X).T  mu(X); mu(X).T 1]
                where mu(X) and Sigma(X) are the mean and variance of the GP posterior at X.
        '''
        mean, var = self.build_predict(X, full_cov=True)
        var = var[:, :, 0] + tf.eye(tf.shape(var)[0], dtype=float_type)*self.likelihood.variance

        # Create omega
        omega = var + tf.matmul(mean, mean, transpose_b=True)
        omega = tf.concat([omega, mean], axis=1)
        omega = tf.concat([omega,
                          tf.concat([tf.transpose(mean), [[1]]], axis=1)],
                          axis=0)

        return omega

    @AutoFlow((float_type, [None, None]))
    def omega(self, X):
        '''
        Just an autoflow wrapper for omega_tf
        '''
        return self.omega_tf(X)

    @AutoFlow((float_type, [None]), (float_type, [None, None]),
              (float_type, [None, None, None]), (float_type, [None, None, None]))
    def acquisition_hessian_tf(self, x, M, dM, domega):
        '''
        Input: x: flattened ndarray [batch_size x self.dim] containing the evaluation points
               M: [(batch_size + 1) x (batch_size + 1)] ndarray
                  containing the solution of the SDP
               dM: [(batch_size + 1) x (batch_size + 1) x (batch_size * self.dim)] ndarray
                  containing the ''jacobian'' of M w.r.t x.
               domega: [(batch_size + 1) x (batch_size + 1) x (batch_size * self.dim)] ndarray
                  containing the ''jacobian'' of the second order moment Omega with respect to x.
        Output: hessian of the acquisition function: [(batch_size * self.dim) x (batch_size * self.dim)] ndarray
        '''
        X = tf.reshape(x, [self.options['batch_size'], -1])
        f = tf.tensordot(self.omega_tf(X), M, axes=2)
        d2f_a = tf.hessians(f, x)[0]
        d2f_b = tf.tensordot(tf.transpose(dM), domega, axes=2)

        return d2f_a + d2f_b

    @AutoFlow((float_type, [None]))
    def domega(self, x):
        '''
        Input: x: flattened ndarray [batch_size x self.dim] containing the evaluation points
        Output: domega: [(batch_size + 1) x (batch_size + 1) x (batch_size * self.dim)] ndarray
                        containing the ''jacobian'' of the second order moment Omega with respect to x.
        '''
        X = tf.reshape(x, [self.options['batch_size'], -1])
        omega = self.omega_tf(X)
        domega = self.jacobian(tf.reshape(omega, [-1]), x)
        return tf.reshape(domega,[omega.shape[0], omega.shape[1], -1])

    def jacobian(self, y_flat, x):
        '''
        Calculates the jacobian of y_flat with respect to x
        Code taken from  jeisses' comment on
        https://github.com/tensorflow/tensorflow/issues/675
        '''
        n = y_flat.shape[0]

        loop_vars = [
            tf.constant(0, tf.int32),
            tf.TensorArray(float_type, size=n),
        ]

        _, jacobian = tf.while_loop(
            lambda j, _: j < n,
            lambda j, result: (j+1, result.write(j, tf.gradients(y_flat[j], x))),
            loop_vars)

        return jacobian.stack()

    def project(self, X):
        '''
        According to Proposition 11, OEI and QEI are not differentiable when the kernel is noiseless and
        there are duplicates in X. In these cases calculating M, SDP's optimizer, becomes increasing hard, as
        the SDP problem becomes ill conditioned.
        
        In this cases, we get around this problem by solving a smaller (projected) SDP problem,
        and providing a subgradient that causes the duplicates to separate and, if requested, a hessian equal to zero.
        Input: X: ndarray [batch_size x self.dim] containing the evaluation points
        Output: 
        If the self.likelihood.variance > 1e-4:
            Returns X (same as input) and V, an identity matrix
        If the kernel is noiseless:
            X_u: ndarray [q x self.dim] containing the unique evaluation points (it removes also duplicates of dataset)
            V: ndarray [q x batch_size] projection matrix that given the solution M of the smaller sdp problem 
                                        can be used to calculate an appropriate subgradient as V.T.dot(M).dot(V)
        '''
        if self.likelihood.variance.value > 1e-4:
            return X, np.eye(X.shape[0] + 1)

        l = self.kern.lengthscales.value

        # Finding duplicates of dataset
        # See https://stackoverflow.com/questions/11903083/find-the-set-difference-between-two-large-arrays-matrices-in-python
        idx = np.where(np.all(np.abs((X/l)[:, None, :] - self.X.value/l) < 1e-2, axis=2))[0]
        idx_ = np.setdiff1d(np.arange(X.shape[0]), idx)
        # Remove duplicates of the dataset
        X_u = X[idx_]
        V = np.eye(X.shape[0])[idx_]
        V[:, idx] = np.random.rand(V.shape[0], len(idx))

        # Remove duplicates in X
        _, idx = np.unique(np.round(X / l, decimals=2), return_index=True, axis=0)
        X_u = X_u[idx]
        V = V[idx]

        V = la.block_diag(V, [[1]])
        # If no points were removed, then just return the original X, to preserve the order of its rows
        if len(X_u) == len(X):
            X_u = X
            V = np.eye(X.shape[0] + 1)

        return X_u, V