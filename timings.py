import GPy
import numpy as np
from methods import QEI, OEI, BO
import GPyOpt.objective_examples
import time
'''
This compares the average time computing time for OEI and QEI
(and their gradients) when performing Bayesian Optimization on 
a standard 5d optimization function (alpine1).
'''


def main():

    options = {'iterations': 1,
               'gp_opt_restarts': 5,
               'acq_opt_restarts': 1,
               'initial_size': 50,
               'normalize_Y': True,
               'noiseless': True,
               'solver': 'SCS',         # for oEI
               'timing': True
               }

    input_dim = 5
    kernel = GPy.kern.RBF(input_dim=input_dim, ARD=True)
    options['kernel'] = kernel

    # Alpine 1 function
    objective = GPyOpt.objective_examples.experimentsNd.alpine1(
        input_dim=input_dim)
    objective.bounds = np.asarray(objective.bounds)
    options['objective'] = objective
    options['job_name'] = 'alpine1_oEI'

    np.random.seed(123)
    X0 = BO.random_sample(options['objective'].bounds, options['initial_size'])
    y0 = options['objective'].f(X0)

    for batch_size in range(2, 18, 1):
        options['batch_size'] = batch_size

        bo_oEI = OEI(options)
        bo_qEI = QEI(options)

        print('----Batch size:', batch_size, '----')
        start = time.time()
        bo_oEI.bayesian_optimization(X0, y0, bo_oEI.objective)
        print('OEI:', np.mean(bo_oEI.timings))
        print('Total time OEI:', time.time() - start)

        start = time.time()
        bo_qEI.bayesian_optimization(X0, y0, bo_qEI.objective)
        print('QEI:', np.mean(bo_qEI.timings))
        print('Total time QEI:', time.time() - start)


if __name__ == "__main__":
    main()
