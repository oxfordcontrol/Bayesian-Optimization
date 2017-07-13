from .bo import BO
import numpy as np
import scipy as sp


class QEI_CL(BO):
    '''
    This class is an implementation of the constant liar heuristic for using
    Expected Improvement in the batch case.

    See:
    Ginsbourger D., Le Riche R., Carraro L. (2010)
    Kriging Is Well-Suited to Parallelize Optimization.
    '''
    def __init__(self, options):
        super(QEI_CL, self).__init__(options)
        # The options dictionary should include the entry 'liar_choice'
        # Its value should be 'min', 'max' or 'mean'
        self.liar_choice = options['liar_choice']

    @classmethod
    def acquisition_fun(cls, X, m):
        '''
        This is simply the EI acqusition function. Code is taken from GPyOpt.
        '''
        ############################################################
        # Copied from GPyOpt/models/gpmodel/predict_withGradients
        # (changed to predict_noiseless)
        ############################################################
        mu, v = m.predict_noiseless(X)
        v = np.clip(v, 1e-10, np.inf)
        s = np.sqrt(v)

        dmudx, dvdx = m.predictive_gradients(X)
        dmudx = dmudx[:, :, 0]
        dsdx = dvdx / (2*np.sqrt(v))
        ###############

        ############################################################
        # Copied from GPyOpt/acqusitions/EI/_compute_acq_withGradients
        ############################################################
        fmin = np.min(m.Y)

        u = (fmin-mu)/s
        phi = np.exp(-0.5 * u**2) / np.sqrt(2*np.pi)
        Phi = 0.5 * sp.special.erfc(-u / np.sqrt(2))

        opt_val = -(fmin - mu) * Phi - s * phi
        dpdX = -dsdx * phi + Phi * dmudx

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
        if self.liar_choice == 'min':
            y_liar = np.min(m.Y).reshape(1, 1)
        elif self.liar_choice == 'max':
            y_liar = np.max(m.Y).reshape(1, 1)
        elif self.liar_choice == 'mean':
            y_liar = np.mean(m.Y).reshape(1, 1)

        # Copy original observations
        X_orig = np.asarray(m.X)
        Y_orig = np.asarray(m.Y)

        # X_final will hold the final choice for the whole batch
        X_final = np.zeros((0, self.dim))
        for i in range(self.batch_size):
            # X will hold the final choice of the i_th point in the batch
            X = None
            # Y will hold the EI of X
            y = None

            # Run local gradient-descent optimizer multiple times
            # to avoid getting stuck in a poor local optimum
            for j in range(self.acq_opt_restarts):
                X0 = self.random_sample(self.bounds, 1)
                res = sp.optimize.minimize(fun=self.acquisition_fun_flat,
                                           x0=X0.flatten(),
                                           args=(m),
                                           method='L-BFGS-B',
                                           jac=True,
                                           bounds=self.bounds,
                                           options=self.optimizer_options)

                '''
                self.derivative_check(
                    lambda X: self.acquisition_fun_flat(X, m), 1
                    )
                '''

                X0 = res.x.reshape(1, self.dim)
                y0 = res.fun[0]

                # Update X if the current local minimum is
                # the best one found so far
                if X is None or y0 < y:
                    X = X0
                    y = y0

            # Don't normalize the function evaluations, since the liar values
            # are calculaled based on m.Y, which is normalized
            # Also, normalizing again would require to retrain the model, but
            # we don't want to do that based on liar values.
            m.set_XY(np.concatenate((m.X, X)), np.concatenate((m.Y, y_liar)))

            # Append i_th choice to the batch
            X_final = np.concatenate((X_final, X))

        # Revert model to the original observations
        m.set_XY(X_orig, Y_orig)

        return X_final
