import numpy as np
import GPy


class gp:
    '''
    This class implements draws from a GP.
    Every time you call the method f, the queried points are saved and added
    along with the generated observations to the latent GPy model.
    '''
    def __init__(self, kernel, bounds, sd=0):
        self.sd = sd
        self.bounds = bounds
        self.kernel = kernel
        self.input_dim = kernel.input_dim
        self.model = None

        # Points generated so far
        self.X = np.zeros((0, self.input_dim))
        self.Y = np.zeros((0, 1))

    def __copy__(self):
        # Copy data
        X = self.X.copy()
        Y = self.Y.copy()
        kernel = self.kernel.copy()
        sd = self.sd
        bounds = self.bounds.copy()

        # Create empty model
        copy = gp(kernel=kernel, bounds=bounds, sd=sd)
        # Fill points queried so far, if any
        if X.size > 0:
            copy.X = X
            copy.Y = Y
            copy.model = GPy.models.GPRegression(
                    X=X, Y=Y, kernel=kernel, noise_var=sd**2
                )

        return copy

    def f(self, X):
        if self.model is None:
            # Get initial samples
            Y = np.random.multivariate_normal(
                np.zeros(X.shape[0]),
                self.kernel.K(X) + np.diag(np.repeat(self.sd**2, X.shape[0]))
            )[:, None]
            self.X = X
            self.Y = Y
            # Initialize GP model
            self.model = GPy.models.GPRegression(
                X=self.X, Y=self.Y, kernel=self.kernel, noise_var=self.sd**2
            )
        else:
            # Get the distribution dictated by the GP for the requested points
            Y_mean, Y_cov = self.model.predict(X, full_cov=True)
            # Sample from the distribution
            Y = np.random.multivariate_normal(Y_mean[:, 0], Y_cov)[:, None]
            self.X = np.concatenate((self.X, X))
            self.Y = np.concatenate((self.Y, Y))
            # Save the observations to the model
            self.model.set_XY(X=self.X, Y=self.Y)

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
