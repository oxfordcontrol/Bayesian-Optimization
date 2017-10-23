import numpy as np
import cvxpy as cvx
from .bo import BO
from gpflow.param import AutoFlow
from gpflow._settings import settings
import tensorflow as tf
float_type = settings.dtypes.float_type


class OEI(BO):
    '''
    This class implements the Optimistic Expected Improvement acquisition function.
    *** THIS IS OUR NOVEL ACQUISITION FUNCTION ***
    '''
    def __init__(self, options):
        super(OEI, self).__init__(options)

    def acquisition(self, x):
        '''
        The acquisition function, supporting sampling
        of the hyperparameters
        '''
        if self.options['samples'] == 0:
            return self.acquisition_no_sample(x)
        else:
            k = x.size // self.dim
            X = x.reshape(k, self.dim)

            N = self.samples.shape[0]
            omegas = np.zeros((k+1, k+1, N))
            for i, s in self.samples.iterrows():
                self.set_parameter_dict(s)
                omegas[:, :, i] = self.omega(X)

            # Solve SDP only once
            fmin = np.min(self.Y.value)
            omega = np.mean(omegas, axis=2)
            M = self.sdp(omega, fmin)[1]

            # Calculate the solution 
            objectives = np.zeros((N))
            gradients = np.zeros((X.size, N))
            for i, s in self.samples.iterrows():
                self.set_parameter_dict(s)
                objectives[i], gradients[:, i] = self.acquisition_tf(X, M)

            return np.asarray([np.mean(objectives, axis=0)]),\
                np.mean(gradients, axis=1)

    def acquisition_no_sample(self, x):
        '''
        The acquisition function when no sampling
        of the hyperparameters is performed
        '''
        k = x.size // self.dim
        X = x.reshape(k, self.dim)
        fmin = np.min(self.Y.value)
        M = self.sdp(self.omega(X), fmin)[1]
        obj, gradient = self.acquisition_tf(X, M)
        return obj, gradient

    @AutoFlow((float_type, [None, None]), (float_type, [None, None]))
    def acquisition_tf(self, X, M):
        '''
        Calculates the acquisition function, given M the optimizer of the SDP.
        The calculation is simply a matrix inner product.
        '''
        fmin = tf.reduce_min(self.Y)
        f = tf.tensordot(self.omega_tf(X), M, axes=2) - fmin
        df = tf.gradients(f, X)[0]
        return tf.reshape(f, [-1]), tf.reshape(df, [-1])

    def omega_tf(self, X):
        '''
        Calculates the second order moment matrix in tensorflow.
        '''
        mean, var = self.likelihood.predict_mean_and_var(*self.build_predict(X, full_cov=True))

        # Create omega
        omega = var[:, :, 0] + tf.matmul(mean, mean, transpose_b=True)
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

    @staticmethod
    def sdp(omega, fmin):
        '''
        Solves the SDP, the solution of which is the acquisiton function.
        Inputs:
            omega: Second order moment matrix
            fmin: min value achieved so far, i.e. min(y0)
        Outpus:
            opt_val: Optimal value of the SDP
            M: Solution of the SDP
            Y: Dual Solution of the SDP
            C: List of auxiliary matrices used in the cone constraints

        The dual formulation is used, as this appears to be faster:

        minimize \sum_{i=0}^{k} <Y_i, C_i> - fmin
        s.t.     Y_i positive semidefinite for all i = 0...k
                 \sum_{i=0}^{k} Y_i = \omega
        '''
        k = omega.shape[1] - 1

        Y = []
        C = []

        C.append(np.zeros((k + 1, k + 1)))
        C[0][-1, -1] = fmin
        Y.append(cvx.Semidef(k+1))
        cost_sum = Y[0]*C[0]

        for i in range(1, k + 1):
            Y.append(cvx.Semidef(k+1))
            C.append(np.zeros((k + 1, k + 1)))
            C[i][-1, i - 1] = 1/2
            C[i][i - 1, -1] = 1/2
            cost_sum += Y[i]*C[i]

        constraints = [sum(Y) == omega]

        objective = cvx.Minimize(cvx.trace(cost_sum))

        prob = cvx.Problem(objective, constraints)
        # Use only one thread for MOSEK
        params = {'MSK_IPAR_NUM_THREADS': 1}
        opt_val = prob.solve(solver=cvx.MOSEK, verbose=False,
                             mosek_params=params) - fmin

        # Assert a valid solution is returned
        assert (isinstance(opt_val, np.ndarray) or isinstance(opt_val, float))\
            and np.isfinite(opt_val)
        
        M = -constraints[0].dual_value
        M = np.asarray((M + M.T)/2)  # From matrix to array

        Y_return = []
        for y in Y:
            Y_return.append(np.asarray(y.value))

        return opt_val, M, Y_return, C
