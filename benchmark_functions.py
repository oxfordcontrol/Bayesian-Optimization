import random
import numpy as np


class scale_function():
    '''
    This wrapper takes another objective and scales its input domain to [0.5,-0.5]^n.
    It's useful to apply when having priors on lengthscales.
    '''
    def __init__(self, function):
        self.bounds = function.bounds.astype(float)
        self.function = function
        self.bounds[:, 0] = -1/2
        self.bounds[:, 1] = 1/2
        if hasattr(function, 'fmin'):
            self.fmin = function.fmin

    def restore(self, X):
        means = (self.function.bounds[:, 1] + self.function.bounds[:, 0])/2
        lengths = self.function.bounds[:, 1] - self.function.bounds[:, 0]

        Xnorm = np.zeros(X.shape)
        for i in range(X.shape[0]):
            Xnorm[i, :] = X[i, :] * lengths + means

        return Xnorm

    def scale(self, X):
        means = (self.function.bounds[:, 1] + self.function.bounds[:, 0])/2
        lengths = self.function.bounds[:, 1] - self.function.bounds[:, 0]

        Xorig = np.zeros(X.shape)
        for i in range(X.shape[0]):
            Xorig[i, :] = (X[i, :] - means) / lengths

        return Xorig

    def f(self, X):
        Xorig = self.restore(X)
        X_ret = ()
        y_ret = ()
        # Split the evaluations into a number of batches
        N = 1
        for i in range(0, len(Xorig), N):
            ret = self.function.f(Xorig[i:i+N])
            if isinstance(ret, tuple):
                y_ret = y_ret + (ret[0],)
                X_ret = X_ret + (ret[1],)
            else:
                y_ret = y_ret + (ret,)

        if isinstance(ret, tuple):
            y_ret = np.concatenate(y_ret)
            X_ret = np.concatenate(X_ret)
            # np.testing.assert_array_almost_equal(self.scale(X_ret), X)
            return y_ret, self.scale(X_ret)
        else:
            y_ret = np.concatenate(y_ret)
            return y_ret


class hart6:
    '''
    Hartmann 6-Dimensional function
    Based on the following MATLAB code:
    https://www.sfu.ca/~ssurjano/hart6.html
    '''
    def __init__(self, sd=0):
        self.sd = sd
        self.bounds = np.array([[0, 1], [0, 1], [0, 1],
                               [0, 1], [0, 1], [0, 1]])
        self.min = np.array([0.20169, 0.150011, 0.476874,
                             0.275332, 0.311652, 0.6573])
        self.fmin = -3.32237

    def f(self, xx):
        if len(xx.shape) == 1:
            xx = xx.reshape((1, 6))

        assert xx.shape[1] == 6

        n = xx.shape[0]
        y = np.zeros(n)
        for i in range(n):
            alpha = np.array([1.0, 1.2, 3.0, 3.2])
            A = np.array([[10, 3, 17, 3.5, 1.7, 8],
                          [0.05, 10, 17, 0.1, 8, 14],
                          [3, 3.5, 1.7, 10, 17, 8],
                          [17, 8, 0.05, 10, 0.1, 14]])
            P = 1e-4 * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                                 [2329, 4135, 8307, 3736, 1004, 9991],
                                 [2348, 1451, 3522, 2883, 3047, 6650],
                                 [4047, 8828, 8732, 5743, 1091, 381]])

            outer = 0
            for ii in range(4):
                inner = 0
                for jj in range(6):
                    xj = xx[i, jj]
                    Aij = A[ii, jj]
                    Pij = P[ii, jj]
                    inner = inner + Aij*(xj-Pij)**2

                new = alpha[ii] * np.exp(-inner)
                outer = outer + new

            y[i] = -outer

        if self.sd == 0:
            noise = np.zeros(n)
        else:
            noise = np.random.normal(0, self.sd, n)

        return (y + noise).reshape((n, 1))

