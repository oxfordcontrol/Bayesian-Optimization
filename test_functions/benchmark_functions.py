from __future__ import print_function
import numpy as np
from gpflow.gpr import GPR
import subprocess
import time
import random
import string
import os
import logging


class scale_function():
    def __init__(self, function):
        self.bounds = function.bounds.astype(float)
        self.function = function
        self.bounds[:, 0] = -1/2
        self.bounds[:, 1] = 1/2
        if hasattr(function, 'fmin'):
            self.fmin = function.fmin

    def denormalize(self, X):
        means = (self.function.bounds[:, 1] + self.function.bounds[:, 0])/2
        lengths = self.function.bounds[:, 1] - self.function.bounds[:, 0]

        Xnorm = np.zeros(X.shape)
        for i in range(X.shape[0]):
            Xnorm[i, :] = X[i, :] * lengths + means

        return Xnorm

    def normalize(self, X):
        means = (self.function.bounds[:, 1] + self.function.bounds[:, 0])/2
        lengths = self.function.bounds[:, 1] - self.function.bounds[:, 0]

        Xorig = np.zeros(X.shape)
        for i in range(X.shape[0]):
            Xorig[i, :] = (X[i, :] - means) / lengths

        return Xorig

    def f(self, X):
        Xorig = self.denormalize(X)
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
            # np.testing.assert_array_almost_equal(self.normalize(X_ret), X)
            return y_ret, self.normalize(X_ret)
        else:
            y_ret = np.concatenate(y_ret)
            return y_ret


class ppo():
    '''
    PPO on OpenAI's roboschool tasks. See https://arxiv.org/pdf/1707.06347.pdf
    Install the PPO implementation from here: https://github.com/nrontsis/baselines
    and roboschool from here: https://github.com/openai/roboschool
    '''
    def __init__(self, experiment='RoboschoolSwimmer-v1'):
        self.bounds = [
            [-5, -3],  # log(stepsize)
            [0.05, 0.5],  # clipping
            [24, 256],   # batch size
            [-3, -1.5],  # log(1-gamma)
            [-2, -1],   # log(1-lam)
        ]
        self.dim = len(self.bounds)
        self.experiment = experiment

    def id_generator(self, size=6,
                     chars=string.ascii_uppercase + string.digits):
        random_id = ''.join(random.choice(chars) for _ in range(size))
        return random_id + '_' + str(os.getpid())

    def f(self, X):
        # Make  batch size integer
        X[:, 2] = np.rint(X[:, 2])
        processes = []
        filenames = []
        for i in range(X.shape[0]):
            x = X[i]

            stepsize = 10.0**(x[0])
            clip_param = x[1]
            batch_size = x[2]
            gamma = 1 - 10.0**(x[3])
            lam = 1 - 10.0**(x[4])


            filenames.append('evals/' + self.id_generator() + '.txt')
            print('clip_par:', clip_param, 'optim_stepsize:', stepsize, 'optim_batchsize:', batch_size, 'gamma:', gamma, 'lam:', lam, 'file:', filenames[i])
            processes.append(subprocess.Popen([
                'python', '-m', 'baselines.ppo1.run_mujoco',
                '--env', self.experiment,
                '--clip-param', str(clip_param),
                '--stepsize', str(stepsize),
                '--batch-size', str(int(batch_size)),
                '--gamma', str(gamma),
                '--lam', str(lam),
                '--num-timesteps', str(int(4e5)),
                '--save=' + (filenames[i])
                ])
            )

        flag = True
        while flag:
            flag = False
            for i in range(len(processes)):
                if processes[i].poll() is None:
                    flag = True
                    time.sleep(1)

        Y = np.zeros((X.shape[0], 1))
        for i in range(len(processes)):
            if os.path.exists(filenames[i]):
                Y[i] = -np.loadtxt(filenames[i])
            else:
                # For failed runs
                logging.getLogger('').critical(filenames[i] + ' call to the black-box function failed')
                Y[i] = 0

        return Y, X


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


class loghart6(hart6):
    '''
    Hartmann 6-Dimensional function
    Based on the following MATLAB code:
    https://www.sfu.ca/~ssurjano/hart6.html
    '''
    def __init__(self, sd=0):
        hart6.__init__(self, sd)
        self.fmin = -np.log(-self.fmin)

    def f(self, xx):
        sd_backup = self.sd

        y = hart6.f(self, xx)
        y = -np.log(-y)

        self.sd = sd_backup

        n = y.shape[0]
        if self.sd == 0:
            noise = np.zeros((n, 1))
        else:
            noise = np.random.normal(0, self.sd, (n, 1))

        return (y + noise).reshape((n, 1))


def peaks(X):
    '''
    MATLAB's peaks function.
    '''
    def __init__(self, sd=0):
        self.sd = sd

    def f(self, X):
        n = X.shape[0]
        Y = np.zeros((X.shape[0], 1))
        Y[:, 0] = 3*(1-X[:, 0])**2 * np.exp(-(X[:, 0]**2) - (X[:, 1]+1)**2) \
            - 10*(X[:, 0]/5 - X[:, 0]**3 - X[:, 1]**5) \
            * np.exp(-X[:, 0]**2 - X[:, 1]**2) \
            - 1/3*np.exp(-(X[:, 0]+1)**2 - X[:, 1]**2)

        if self.sd == 0:
            noise = np.zeros(n)
        else:
            noise = np.random.normal(0, self.sd, n)

        return (Y + noise).reshape((n, 1))
