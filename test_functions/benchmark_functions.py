import numpy as np
from gpflow.gpr import GPR


class gp:
    '''
    This class implements draws from a GP.
    Every time you call the method f, the queried points are saved and added
    along with the generated observations to the latent gpflow model.
    '''
    def __init__(self, kernel, bounds, X=None, Y=None,
                 sd=0, mean_function=None):
        self.bounds = bounds.copy()
        self.sd = sd
        self.input_dim = kernel.input_dim
        if X is None or Y is None:
            X = np.zeros((0, self.input_dim))
            Y = np.zeros((0, 1))

        self.model = GPR(
            X=X.copy(),
            Y=Y.copy(),
            kern=kernel,
            mean_function=mean_function
        )
        self.model.likelihood.variance = self.sd**2

    def __copy__(self):
        copy = gp(kernel=self.model.kern,
                  bounds=self.bounds,
                  X=self.model.X.value,
                  Y=self.model.Y.value,
                  sd=self.sd,
                  mean_function=self.model.mean_function)

        return copy

    def f(self, X):
        # Get the distribution dictated by the GP for the requested points
        Y_mean, Y_cov = self.model.predict_f_full_cov(X)
        # Sample from the distribution
        Y = np.random.multivariate_normal(Y_mean[..., 0], Y_cov[..., 0])[:, None]
        # Save the observations to the model
        self.model.X = np.concatenate((self.model.X.value, X))
        self.model.Y = np.concatenate((self.model.Y.value, Y))

        return Y


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
        self.fmin = -3.04245774

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

            y[i] = -(2.58 + outer) / 1.94

        if self.sd == 0:
            noise = np.zeros(n)
        else:
            noise = np.random.normal(0, self.sd, n)

        return (y + noise).reshape((n, 1))


class loghart6:
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
        self.fmin = -np.log(-3.04245774)

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

            y[i] = -(2.58 + outer) / 1.94

        # Apply log transform (see DiceKriging, DiceOptim: Two R
        # Packages for the Analysis of Computer Experiments by
        # Kriging-Based Metamodeling and Optimization)
        y = -np.log(-y)

        if self.sd == 0:
            noise = np.zeros(n)
        else:
            noise = np.random.normal(0, self.sd, n)

        return (y + noise).reshape((n, 1))
