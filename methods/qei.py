from .bo import BO
import GPy
import numpy as np
import scipy as sp
import rpy2.robjects as robjects        # This initializes R
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
import os
import time
import pdb


class QEI(BO):
    '''
    This class is a wrapper for the R implementation of the multipoint
    Expected Improvement acquisition function and its gradient.
    It uses the rpy2 package and some R auxiliary wrapper functions
    defined in r_callers.R

    Fast Computation of the Multi-Points Expected Improvement with Applications
    in Batch Selection
    http://dx.doi.org/10.1007/978-3-642-44973-4_7

    Differentiating the Multipoint Expected Improvement for
    Optimal Batch Design
    http://dx.doi.org/10.1007/978-3-319-27926-8_4

    Dicekriging, Diceoptim: Two R packages for the analysis of computer
    experiments by kriging-based metamodeling and optimization.
    https://www.jstatsoft.org/article/view/v051i0
    '''
    def __init__(self, options):
        super(QEI, self).__init__(options)

        importr('DiceOptim')  # Import DiceOptim R package
        # Enables the conversion of numpy objects to rpy2 objects
        numpy2ri.activate()

        # Include the file r_callers to the R path
        dir_path = os.path.dirname(os.path.realpath(__file__))
        robjects.r("source('" + dir_path + "/r_callers.R')")

    def acquisition_fun(self, X, m):
        # Pack the parameters of the model in the format
        # required by DiceKriging
        if m.kern.ARD:
            cov_param = np.asarray(m.kern.lengthscale)
        else:
            cov_param = np.repeat(m.kern.lengthscale, X.shape[1])
        cov_var = np.asarray(m.kern.variance)
        var = np.asarray(m.likelihood.variance)

        if isinstance(m.kern, GPy.kern.RBF):
            cov_type = 'gauss'  # DiceKriging calls the RBF kernel 'gauss'
        else:
            print('R caller not implemented for the kernel you used')
            assert False

        # Call R. Warning: the code in qEI_caller calculates the noiseless
        # expected improvement
        qEI = robjects.r['qEI_caller'](X, m.X, m.Y, cov_type, cov_param,
                                       cov_var, var)
        grad = robjects.r['qEI_grad_caller'](X, m.X, m.Y, cov_type,
                                             cov_param, cov_var, var)
        if self.timing:
            r_time = robjects.r['qEI_timing'](X, m.X, m.Y, cov_type,
                                              cov_param, cov_var, var)[0]
            self.timings.append(r_time)

        opt_val = -np.asarray(qEI)
        dpdX = -np.array(grad)

        '''
        --------------------------
        Sanity checks
        --------------------------
        # Test that DiceOptim and GPy predictions are the same
        for i in range(1000):
            X_test = self.random_sample(self.bounds, 5)
            mu_R = robjects.r['model_mean'](X_test, m.X, m.Y, cov_type,
                                                   cov_param, cov_var, var)
            vr_R = robjects.r['model_var'](X_test, m.X, m.Y, cov_type,
                                                   cov_param, cov_var, var)
            mu, vr = m.predict_noiseless(X_test)
            print('error mean:', np.linalg.norm(mu[0] - np.asarray(mu_R))/
                  np.linalg.norm(mu[0])*100, '%')
            print('error var:', np.linalg.norm(vr[0] - np.asarray(vr_R))/
                  np.linalg.norm(vr[0])*100, '%')
            pdb.set_trace()

        # Compare the multipoint expected improvement given by DiceOptim
        # against sampling. Warning: sometimes qEI is arbitrarily wrong see
        # qEI_problem.R
        mean, cov = m.predict_noiseless(X, full_cov=True)
        eta = min(m.Y)
        N = 100000
        imp = np.zeros(N)
        for k in range(N):
            Y = np.random.multivariate_normal(mean[:, 0], cov)[:, None]
            imp[k] = min(eta, min(Y)) - eta

        print('EI via sampling:', np.mean(imp))
        print('EI via DiceOptim:', opt_val)
        '''

        return opt_val, dpdX

    def acquisition_fun_flat(self, X, m):
        '''
        Wrapper for acquisition_fun, where X is considered as a vector
        '''
        start = time.time()
        n = m.X.shape[1]
        k = X.shape[0]//n

        (opt_val, dpdX) = self.acquisition_fun(X.reshape(k, n), m)
        end = time.time()
        self.timings.append(end - start)
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

            res = sp.optimize.minimize(fun=self.acquisition_fun_flat,
                                       x0=X0.flatten(),
                                       args=(m),
                                       method='L-BFGS-B',
                                       jac=True,
                                       bounds=bounds_tiled,
                                       options=self.optimizer_options
                                       )
            '''
            self.derivative_check(
                lambda X: self.acquisition_fun_flat(X, m), self.batch_size
                )
            '''
            X0 = res.x.reshape(self.batch_size, self.dim)
            y0 = res.fun[0]

            # Update X if the current local minimum is
            # the best one found so far
            if X is None or y0 < y:
                X = X0
                y = y0

        # Assert that at least one optimization run succesfully
        assert X is not None

        return X
