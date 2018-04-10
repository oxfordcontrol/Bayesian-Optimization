import numpy as np
from methods.oei import OEI
import gpflow
import sys
sys.path.append('..')
from benchmark_functions import scale_function, hart6


def create_model(batch_size=2):
    options = {}
    options['samples'] = 0
    options['priors'] = 0
    options['batch_size'] = batch_size
    options['iterations'] = 5
    options['opt_restarts'] = 2
    options['initial_size'] = 10
    options['model_restarts'] = 10
    options['normalize_Y'] = 1
    options['noise'] = 1e-6
    options['nl_solver'] = 'bfgs'
    options['hessian'] = True

    options['objective'] = hart6()
    options['objective'].bounds = np.asarray(options['objective'].bounds)
    options['objective'] = scale_function(options['objective'])

    input_dim = options['objective'].bounds.shape[0]
    options['kernel'] = gpflow.kernels.Matern32(
            input_dim=input_dim, ARD=False
        )

    options['job_name'] = 'tmp'

    bo = OEI(options)
    # Initialize
    bo.bayesian_optimization()

    return bo
