from .bo import BO
import GPy
import numpy as np
import scipy as sp
import cvxpy as cvx


class OEI(BO):
    '''
    This class implements the Optimistic Expected Improvement acquisition
    function.

    See:
    N Rontsis, MA Osborne, PJ Goulart
    Distributionally Robust Optimization Techniques in Batch Bayesian
    Optimization
    https://arxiv.org/abs/1707.04191
    '''
    def __init__(self, options):
        super(OEI, self).__init__(options)

    @classmethod
    def acquisition_fun(cls, X, m):
        # The acquisition function is simply the solution of an SDP
        eta = min(m.Y)
        (opt_val, M) = cls.sdp(cls.Omega_(X, m), eta)

        # The derivative of the acquisition function p is calculated
        # by providing dp/dK(X,X0) and dp/dK(X,X) to the GPy function
        # m.kern.gradients_X, which performs the chain rule to get
        # dp/dX.
        dpdX = \
            m.kern.gradients_X(cls.dpdK0(m, X, M), X, m.X) \
            + m.kern.gradients_X(cls.dpdK(m, X, M), X)

        return opt_val, dpdX

    @classmethod
    def acquisition_fun_flat(cls, X, m):
        '''
        Wrapper for acquisition_fun, where X is considered as a vector
        '''
        n = m.X.shape[1]
        k = X.shape[0]//n

        (opt_val, dpdX) = cls.acquisition_fun(X.reshape(k, n), m)
        return opt_val, dpdX.flatten()

    def acq_fun_optimizer(self, m):
        X = None    # Will hold the final choice
        y = None    # Will hold the expected improvement of the final choice

        # Run local gradient-descent optimizer multiple times
        # to avoid getting stuck in a poor local optimum
        for j in range(self.acq_opt_restarts):
            # Initial point of the optimization
            X0 = self.random_sample(self.bounds, self.batch_size)
            # Tile bounds to match batch size
            bounds_tiled = np.tile(self.bounds, (self.batch_size, 1))

            try:
                res = sp.optimize.minimize(fun=self.acquisition_fun_flat,
                                           x0=X0.flatten(),
                                           args=(m),
                                           method='L-BFGS-B',
                                           jac=True,
                                           bounds=bounds_tiled,
                                           options=self.optimizer_options
                                           )
                X0 = res.x.reshape(self.batch_size, self.dim)
                y0 = res.fun[0]
                # Update X if the current local minimum is
                # the best one found so far
                if X is None or y0 < y:
                    X = X0
                    y = y0
            except AssertionError:
                print('Solver failed. Omega is close to singular.')

            '''
            self.derivative_check(
                lambda X: self.acquisition_fun_flat(X, m), self.batch_size
                )
            '''

        # Assert that at least one optimization run succesfully
        assert X is not None

        return X

    @staticmethod
    def sdp(Omega, eta):
        '''
        Solves the SDP, the solution of which is the acquisiton function.
        Inputs:
            Omega: Second order moment matrix
            eta: min value achieved so far, i.e. min(y0)
        Outpus:
            opt_val: Optimal value of the SDP
            M: Optimizer of the SDP

        The dual formulation is used, as this appears to be faster:

        minimize \sum_{i=0}^{k} <Y_i, C_i> - eta
        s.t.     Y_i positive semidefinite for all i = 0...k
                 \sum_{i=0}^{k} Y_i = \Omega
        '''
        k = Omega.shape[1] - 1

        Y = []
        C = []

        C.append(np.zeros((k + 1, k + 1)))
        C[0][-1, -1] = eta
        Y.append(cvx.Semidef(k+1))
        cost_sum = Y[0]*C[0]

        for i in range(1, k + 1):
            Y.append(cvx.Semidef(k+1))
            C.append(np.zeros((k + 1, k + 1)))
            C[i][-1, i - 1] = 1/2
            C[i][i - 1, -1] = 1/2
            cost_sum += Y[i]*C[i]

        constraints = [sum(Y) == Omega]

        objective = cvx.Minimize(cvx.trace(cost_sum))

        prob = cvx.Problem(objective, constraints)
        params = {'MSK_IPAR_NUM_THREADS': 1}
        opt_val = prob.solve(solver=cvx.MOSEK, verbose=False,
                             mosek_params=params) - eta

        assert isinstance(opt_val, np.ndarray) and np.isfinite(opt_val)

        M = -constraints[0].dual_value
        M = np.asarray((M + M.T)/2)  # From matrix to array

        return opt_val, M

    @staticmethod
    def Omega_(X, m):
        '''
        Calculation of the second order moment matrix
        '''
        k = X.shape[0]

        mu, var = m.predict_noiseless(X, full_cov=True)
        O = np.zeros((k + 1, k + 1))
        O[:-1, :-1] = var + mu.dot(mu.T)
        O[None, -1, :-1] = mu.T
        O[:-1, -1, None] = mu
        O[-1, -1] = 1
        return O

    @staticmethod
    def dpdK0(m, X, M):
        '''
        Returns the partial derivative of the acquistion function with
        respect to K(X, X0), where X is the batch choice and X0 the
        points already in the dataset of the GP model.

        Assume that K = K(X,X), K0 = K(X0, X), K00 = K(X0, X0) + sigma*I, and
        M = [M11 m12; m12' m22] the optimizer of the SDP.
        Then, the partial derivative is given by:
        dp/dK0' = <dOmega/dK0', M> =
        -2(((K00\y0)*(K00\y0)'-inv(K00))*K0*M11 + 2*K00\y0*m12')'

        The following code was written by combining pieces of code from GPy.
        It avoids the calculation of inv(K00).
        '''
        M11 = M[:-1, :-1]
        m12 = M[:-1, -1, None]

        K0 = m.kern.K(m.X, X)
        # Cholesky Decomposition of the Woodbury matrix
        # The inverse of the woodbury matrix, in the gaussian likelihood
        # case is defined as (K_{xx} + \Sigma_{xx})^{-1}
        W_c = m.posterior.woodbury_chol
        # Next line was copied from  m.posterior.woodbury_vector()
        tmp = GPy.util.linalg.dpotrs(W_c, K0)[0]

        # Woodbury Vector i.e.
        # (K_{xx} + \Sigma_{xx})^{-1}*Y
        W_v = m.posterior.woodbury_vector

        # Combine all the elements and transpose the result to get the
        # derivative w.r.t. K0.T = K(X, X0)
        return (-2*(tmp - W_v.dot(W_v.T).dot(K0)).dot(M11) +
                2*W_v.dot(m12.T)).T

    @staticmethod
    def dpdK(m, X, M):
        '''
        Returns the partial derivative of the acquistion function with
        respect to K(X, X), where X is the batch choice.
        '''
        M11 = M[:-1, :-1]

        return M11
