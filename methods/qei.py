from __future__ import print_function
from .bo import BO
import gpflow
import numpy as np
import rpy2.robjects as robjects        # This initializes R
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
import os


class QEI(BO):
    '''
    This class is a wrapper for the R implementation of the multipoint
    Expected Improvement acquisition function and its gradient.
    It uses the rpy2 package and some R auxiliary wrapper functions
    defined in r_callers.R

    Some errors were fixed for batch sizes of 1, and 1d cases, so please
    install the following modified version:
    https://github.com/nrontsis/DiceOptim/tree/62e72c78868bde14e13cb6227bd44207302b84b6

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
        self.r_model = None

    def update_r_model(self):
        # Pack the parameters of the model in the format
        # required by DiceKriging
        if self.kern.ARD:
            cov_param = np.asarray(self.kern.lengthscales.value)
        else:
            cov_param = np.repeat(self.kern.lengthscales.value, self.dim)
        cov_var = np.asarray(self.kern.variance.value)
        var = np.asarray(self.likelihood.variance.value)

        if isinstance(self.kern, gpflow.kernels.RBF):
            cov_type = 'gauss'  # DiceKriging calls the RBF kernel 'gauss'
        elif isinstance(self.kern, gpflow.kernels.Matern52):
            cov_type = 'matern5_2'
        elif isinstance(self.kern, gpflow.kernels.Matern32):
            cov_type = 'matern3_2'
        else:
            print('R caller not implemented for the kernel you used')
            assert False

        # Call R. Warning: the code in qEI_caller calculates the noiseless
        # expected improvement
        if self.options['mean_function'] is None:
            self.r_model = robjects.r['create_model'](
                self.X.value, self.Y.value,
                cov_type, cov_param, cov_var, var,
                self.options['noise'] is not None,
                self.options['ard'] == 1
            )
        else:
            # Please make sure the mean function is a quadratic
            A = self.options['mean_function'].A.value
            # Convert to R trend parameters (see r_callers.R)
            A = A.dot(A.T)
            # Make sure we're in 1d case
            assert A.size == 1
            self.r_model = robjects.r['create_model_1d_quadratic_mean'](
                self.X.value, self.Y.value,
                cov_type, cov_param, cov_var, var,
                A[:]
            )

    def get_suggestion(self, batch_size):
        self.update_r_model()
        return super(QEI, self).get_suggestion(batch_size)

    def acquisition_tf(self, X):
        if self.r_model is None:
            self.update_r_model()

        qEI = robjects.r['qEI_caller'](X, self.r_model)
        grad = robjects.r['qEI_grad_caller'](X, self.r_model)
        opt_val = -np.array(qEI)
        dpdX = -np.array(grad)

        '''
        --------------------------
        Sanity checks
        --------------------------
        # Test that DiceOptim and GPy predictions are the same
        for i in range(1000):
            X_test = self.random_sample(self.bounds, 2)
            mu_R = robjects.r['model_mean'](X_test, r_model)
            vr_R = robjects.r['model_var'](X_test, r_model)
            mu, vr = self.predict_f(X_test)
            print('error mean:', np.linalg.norm(mu[:, 0] - np.asarray(mu_R))/
                  np.linalg.norm(mu[:, 0])*100, '%')
            print('error var:', np.linalg.norm(vr[:, 0] - np.asarray(vr_R))/
                  np.linalg.norm(vr[:, 0])*100, '%')
            pdb.set_trace()

        # Compare the multipoint expected improvement given by DiceOptim
        # against sampling. Warning: sometimes qEI is arbitrarily wrong see
        # qEI_problem.R
        mean, cov = self.predict_f_full_cov(X)
        fmin = min(self.Y.value)
        N = 100000
        imp = np.zeros(N)
        for k in range(N):
            Y = np.random.multivariate_normal(mean[:, 0], cov[:, :, 0])[:, None]
            imp[k] = min(fmin, min(Y)) - fmin

        print('EI via sampling:', np.mean(imp))
        print('EI via DiceOptim:', opt_val)
        pdb.set_trace()
        '''

        return opt_val, dpdX.flatten()
