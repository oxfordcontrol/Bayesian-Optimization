import GPy
import numpy as np
from methods import QEI, OEI, BO
import GPyOpt.objective_examples
from test_functions import benchmark_functions
import time
import os
'''
This compares the average time computing time for OEI and QEI
(and their gradients) when performing Bayesian Optimization on 
a standard 5d optimization function (alpine1).
'''


def main():
    # os.environ["OMP_NUM_THREADS"] = "4"

    options = {
               # Run only the first iteration,
               # as the experiments takes a lot of time
               'iterations': 1,
               'gp_opt_restarts': 20,  #
               'acq_opt_restarts': 10,
               'initial_size': 20,
               'normalize_Y': True,
               'noiseless': True,
               'solver': 'SCS',         # for oEI
               'timing': True
               }

    input_dim = 6
    kernel = GPy.kern.RBF(input_dim=input_dim, ARD=True)
    options['kernel'] = kernel

    # Loghart6 function
    '''
    objective = GPyOpt.objective_examples.experimentsNd.alpine1(
        input_dim=input_dim)
    objective.bounds = np.asarray(objective.bounds)
    '''
    objective = benchmark_functions.loghart6()
    objective.bounds = np.asarray(objective.bounds)
    options['objective'] = objective
    options['job_name'] = 'loghart6'

    '''
    Run once before to avoid initialization timings in the benchmark
    '''
    options['batch_size'] = 2
    X0 = BO.random_sample(options['objective'].bounds, options['initial_size'])
    y0 = options['objective'].f(X0)
    bo_oEI = OEI(options)
    bo_qEI = QEI(options)
    bo_oEI.bayesian_optimization(X0, y0, bo_oEI.objective)
    bo_qEI.bayesian_optimization(X0, y0, bo_qEI.objective)

    '''
    Actual benchmark
    '''
    sizes = np.array([2, 3, 6, 10, 20, 40])
    for batch_size in sizes:
        options['batch_size'] = batch_size

        bo_oEI = OEI(options)
        bo_qEI = QEI(options)

        # Set the seed (same for OEI and QEI)
        np.random.seed(123)
        X0 = BO.random_sample(options['objective'].bounds, options['initial_size'])
        y0 = options['objective'].f(X0)

        print('----Batch size:', batch_size, '----')
        start = time.time()
        bo_oEI.bayesian_optimization(X0, y0, bo_oEI.objective)
        oei_timing = np.mean(bo_oEI.timings)

        if batch_size < 20:
            # Set the seed (same for OEI and QEI)
            np.random.seed(123)
            X0 = BO.random_sample(options['objective'].bounds, options['initial_size'])
            y0 = options['objective'].f(X0)

            bo_qEI.bayesian_optimization(X0, y0, bo_qEI.objective)
            qei_timing = np.mean(bo_qEI.timings)
            print('OEI:', "%0.4f" % oei_timing, 'QEI:', "%0.4f" % qei_timing)
            print('Ratio:', "%0.2f" % (qei_timing/oei_timing))
        else:
            print('OEI:', "%0.4f" % oei_timing)
            print('Iteration time:', "%0.2f" % (time.time() - start))


if __name__ == "__main__":
    main()
