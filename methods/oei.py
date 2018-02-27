from __future__ import print_function
import numpy as np
from .bo import BO
from gpflow.param import AutoFlow
from gpflow._settings import settings
import tensorflow as tf
import scipy.linalg as la
import logging
from .sdps import sdp
float_type = settings.dtypes.float_type


class OEI(BO):
    '''
    This class implements the Optimistic Expected Improvement acquisition function.
    *** THIS IS OUR NOVEL ACQUISITION FUNCTION ***
    '''
    def __init__(self, options):
        super(OEI, self).__init__(options)

    def acquisition(self, x, fmin=None):
        '''
        The acquisition function, supporting sampling
        of the hyperparameters
        '''
        if fmin is None:
            fmin = np.min(self.predict_f(self.X.value)[0])

        x = x.flatten()  # Make sure x is flat

        if self.options['samples'] == 0:
            return self.acquisition_no_sample(x, fmin)

        X = x.reshape((-1, self.dim))
        k = x.size // self.dim

        N = self.samples.shape[0]
        omegas = np.zeros((k+1, k+1, N))
        for i, s in self.samples.iterrows():
            self.set_parameter_dict(s)
            omegas[:, :, i] = self.omega(X)

        # Solve SDP only once
        omega = np.mean(omegas, axis=2)
        M = sdp(omega, fmin)[1]

        # Calculate the solution
        objectives = np.zeros((N))
        gradients = np.zeros((x.size, N))
        for i, s in self.samples.iterrows():
            self.set_parameter_dict(s)
            objectives[i], gradients[:, i] = self.acquisition_tf(X, M)

        return np.asarray([np.mean(objectives, axis=0)]),\
            np.mean(gradients, axis=1)

    def acquisition_no_sample(self, x, fmin=None):
        '''
        The acquisition function when no sampling
        of the hyperparameters is performed
        '''
        if fmin is None:
            fmin = np.min(self.predict_f(self.X.value)[0])

        X = x.reshape((-1, self.dim))

        M = sdp(self.omega(X), fmin)[1]
        obj, gradient = self.acquisition_tf(X, M)
        return obj, gradient

    @AutoFlow((float_type, [None, None]), (float_type, [None, None]))
    def acquisition_tf(self, X, M):
        '''
        Calculates the acquisition function, given M the optimizer of the SDP.
        The calculation is simply a matrix inner product.
        '''
        f = tf.tensordot(self.omega_tf(X), M, axes=2)
        df = tf.gradients(f, X)[0]
        return tf.reshape(f, [-1]), tf.reshape(df, [-1])

    def omega_tf(self, X):
        '''
        Calculates the second order moment matrix in tensorflow.
        '''
        mean, var = self.likelihood.predict_mean_and_var(
            *self.build_predict(X, full_cov=True)
        )

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
    def P_(k):
        A = np.zeros((0, k*(k+1)//2))
        for i in range(k):
            B = np.zeros((0, 0))
            for j in range(i):
                tmp = np.zeros((1, k - j))
                tmp[0, -(k - i)] = 1
                B = la.block_diag(B, tmp)

            B = la.block_diag(B, np.eye(k - i))
            B = np.hstack((B, np.zeros((k, k*(k+1)//2 - B.shape[1]))))

            A = np.vstack((A, B))

        return A

    @staticmethod
    def P(k):
        A = np.zeros((0, 0))
        I = np.eye(k)
        for i in range(k):
            A = la.block_diag(A, I[i:, :])

        return A

    @staticmethod
    def solve(LU, C):
        assert np.allclose(C, C.T)
        n = C.shape[0]

        c = np.zeros(len(LU[1]))
        c[-n*(n+1)//2:] = C[np.triu_indices(n)]
        x = la.lu_solve(LU, c)

        def create_matrix(x):
            A = np.zeros((n, n))
            A[np.triu_indices(n)] = x[-n*(n+1)//2:]
            A = A + A.T
            A[np.diag_indices(n)] = A[np.diag_indices(n)]/2
            return A

        q = n*(n+1)//2
        dY = []
        for i in range(0, x.size - q, q):
            dY.append(create_matrix(x[i:i+q]))

        dM = create_matrix(x[-n*(n+1)//2:])

        return dM, dY

    def factor(self, omega):
        k = omega.shape[0] - 1
        # The acquisition function is simply the solution of an SDP
        opt_val, M_, Y_, C_ = sdp(omega, np.min(self.Y.value))

        # Calculate dM analytically
        S_ = C_ - M_

        P = self.P(k + 1)
        P_ = self.P_(k + 1)

        A1, A2 = None, None
        for i in range(k + 1):
            e, U = la.eigh(S_[i])
            U = U.T
            SS = np.kron((S_[i].dot(U)).T, U.T)
            YY = np.kron(U.T, U.T.dot(Y_[i]))

            if A1 is not None:
                A1 = la.block_diag(A1, P.dot(SS).dot(P_))
                A2 = np.vstack((A2, P.dot(YY).dot(P_)))
            else:
                A1 = P.dot(SS).dot(P_)
                A2 = P.dot(YY).dot(P_)

        A3 = np.tile(np.eye((k+1)*(k+2)//2), k+1)
        A4 = np.zeros((A3.shape[0], A1.shape[1] + A2.shape[1] - A3.shape[1]))

        A = np.vstack((
            np.hstack((A1, -A2)),
            np.hstack((A3, A4))
        ))
        # print('New Condition:', np.linalg.cond(A))

        LU = la.lu_factor(A)

        return LU

    def acquisition_hessian(self, x):
        '''
        The hessian of the acquisition function, supporting sampling
        of the hyperparameters
        '''
        assert self.options['samples'] == 0

        X = x.reshape((self.batch_size, -1))

        omega = self.omega(X)
        fmin = np.min(self.Y.value)
        M = sdp(omega, fmin)[1]

        LU = self.factor(omega)

        domega = self.domega(X)
        dM = np.zeros(M.shape + (x.size,))
        # We could possibly get rid of (or compute in parallel) this for loop
        for i in range(x.size):
            dM[:, :, i], _ = self.solve(LU, domega[:, :, i])

        return self.acquisition_hessian_tf(x, M, dM)

    @AutoFlow((float_type, [None]), (float_type, [None, None]),
              (float_type, [None, None, None]))
    def acquisition_hessian_tf(self, x, M, dM):
        X = tf.reshape(x, [self.options['batch_size'], -1])
        f = tf.tensordot(self.omega_tf(X), M, axes=2)
        d2f_a = tf.hessians(f, x)[0]

        d2f_b = tf.tensordot(tf.transpose(dM), self.domega_tf(X), axes=2)

        return d2f_a + d2f_b

    def domega_tf(self, X):
        omega = self.omega_tf(X)
        domega_list = []
        for i in range(self.options['batch_size'] + 1):
            tmp_row = []
            for j in range(self.options['batch_size'] + 1):
                if i >= j:
                    # If its an upper triangular element then
                    # Calculate the gradients
                    if i == j:
                        # Scale the diagonal elements
                        tmp_row.append(tf.reshape(
                            tf.gradients(0.5*omega[i, j], X)[0],
                            [-1]
                        ))
                    else:
                        tmp_row.append(tf.reshape(
                            tf.gradients(omega[i, j], X)[0],
                            [-1]
                        ))
                else:
                    # Otherwise, set them to zero
                    tmp_row.append(0*tmp_row[0])

            domega_list.append(tmp_row)
        # Convert to tensor
        domega = tf.stack(domega_list)
        # Calculate the final tensor
        # (until now the lower triangular elements were zero)
        domega = domega + tf.transpose(domega, [1, 0, 2])

        return domega

    @AutoFlow((float_type, [None, None]))
    def domega(self, X):
        return self.domega_tf(X)