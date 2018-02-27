import numpy as np
import methods
import gpflow
import GPyOpt


def create_model(batch_size=2):
    options = {}
    options['samples'] = 0
    options['priors'] = 0
    options['batch_size'] = batch_size
    options['iterations'] = 0
    options['opt_restarts'] = 5
    options['initial_size'] = 10
    options['model_restarts'] = 20
    options['normalize_Y'] = 1
    options['noise'] = 1e-6
    options['nl_solver'] = 'scipy'
    options['hessian'] = True

    options['objective'] = GPyOpt.objective_examples.experiments2d.branin() 
    options['objective'].bounds = np.asarray(options['objective'].bounds)

    input_dim = options['objective'].bounds.shape[0]
    options['kernel'] = gpflow.kernels.RBF(
            input_dim=input_dim, ARD=True
        )

    options['job_name'] = 'tmp'

    bo = methods.OEI(options)
    # Initialize
    bo.bayesian_optimization()

    return bo
